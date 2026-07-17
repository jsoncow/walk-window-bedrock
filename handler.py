"""
walk-planner: proposes two optimal 30-minute walk windows for tomorrow.

Runs the evening before, so you can plan the morning. Triggered by EventBridge
Scheduler. Fetches tomorrow's hourly forecast from Open-Meteo, scores each
half-hour slot deterministically, picks the two best non-adjacent windows, asks
Bedrock to write the rationale, and publishes to SNS.

No external dependencies -- uses urllib so the function can be deployed as a
plain zip with no layer.
"""

import datetime as dt
import json
import logging
import os
import urllib.parse
import urllib.request

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

sns = boto3.client("sns")
bedrock = boto3.client("bedrock-runtime")

# --- Configuration (all from environment variables) ------------------------

LAT = float(os.environ.get("LATITUDE", "35.7796"))
LON = float(os.environ.get("LONGITUDE", "-78.6382"))
TIMEZONE = os.environ.get("TIMEZONE", "America/New_York")
SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")

# Earliest / latest hour the user is willing to walk (local time, 24h).
WINDOW_START_HOUR = int(os.environ.get("WINDOW_START_HOUR", "6"))
WINDOW_END_HOUR = int(os.environ.get("WINDOW_END_HOUR", "20"))

# Comfort band for apparent temperature, in Fahrenheit.
IDEAL_TEMP_MIN = float(os.environ.get("IDEAL_TEMP_MIN", "50"))
IDEAL_TEMP_MAX = float(os.environ.get("IDEAL_TEMP_MAX", "72"))

# Minimum gap between the two proposals, so they read as real alternatives.
MIN_SEPARATION_MINUTES = int(os.environ.get("MIN_SEPARATION_MINUTES", "90"))

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation_probability",
    "wind_speed_10m",
    "uv_index",
]


# --- Fetch -----------------------------------------------------------------


def fetch_forecast():
    """
    Pull today and tomorrow's hourly forecast plus sunrise/sunset, in the
    location's local time. Two days, because the run happens tonight but every
    slot we care about is tomorrow.
    """
    query = urllib.parse.urlencode(
        {
            "latitude": LAT,
            "longitude": LON,
            "hourly": ",".join(HOURLY_VARS),
            "daily": "sunrise,sunset",
            "timezone": TIMEZONE,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_days": 2,
        }
    )
    req = urllib.request.Request(
        f"{FORECAST_URL}?{query}", headers={"User-Agent": "walk-planner/1.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Open-Meteo returned HTTP {resp.status}")
        return json.loads(resp.read())


# --- Slot construction -----------------------------------------------------


def target_day(forecast):
    """
    Tomorrow, as the forecast location reckons it.

    Deliberately read out of the API response rather than computed here. Lambda
    runs in UTC, so a 20:00 America/New_York trigger is already 00:00 UTC the
    next day -- date.today() + 1 would quietly plan for the day after tomorrow,
    and only during standard time, which is a delightful bug to chase in March.
    Open-Meteo reports in the location's timezone, so daily index 1 is
    unambiguous no matter when the function fires.
    """
    return dt.date.fromisoformat(forecast["daily"]["time"][1])


def build_slots(forecast, target):
    """
    Expand tomorrow's hourly forecast into 30-minute slots.

    Open-Meteo reports on the hour, so the :30 slot is a linear interpolation
    between that hour and the next. Good enough for walk planning, and it stops
    every suggestion from landing awkwardly on the hour.
    """
    hourly = forecast["hourly"]
    times = [dt.datetime.fromisoformat(t) for t in hourly["time"]]

    # Index 1 = tomorrow. Today's sunrise/sunset would be off by a minute or
    # two, which sounds harmless until it trims a slot at the edge of dawn.
    sunrise = dt.datetime.fromisoformat(forecast["daily"]["sunrise"][1])
    sunset = dt.datetime.fromisoformat(forecast["daily"]["sunset"][1])

    slots = []
    for i in range(len(times) - 1):
        for half in (0, 30):
            start = times[i] + dt.timedelta(minutes=half)
            end = start + dt.timedelta(minutes=30)

            # Tomorrow only -- the response also carries today's hours.
            if start.date() != target:
                continue
            # Daylight only, and inside the user's stated availability.
            if start < sunrise or end > sunset:
                continue
            if not (WINDOW_START_HOUR <= start.hour < WINDOW_END_HOUR):
                continue

            weight = half / 60.0
            conditions = {
                var: _lerp(hourly[var][i], hourly[var][i + 1], weight)
                for var in HOURLY_VARS
            }
            slots.append({"start": start, "conditions": conditions})

    return slots


def _lerp(a, b, weight):
    if a is None or b is None:
        return a if a is not None else b
    return a + (b - a) * weight


# --- Scoring ---------------------------------------------------------------


def score_slot(slot):
    """
    Score a slot 0-100. Deterministic on purpose: this is the part that must
    never hallucinate a dry window into an afternoon thunderstorm.
    """
    c = slot["conditions"]
    score = 100.0
    reasons = []

    feels = c["apparent_temperature"]
    if feels < IDEAL_TEMP_MIN:
        penalty = (IDEAL_TEMP_MIN - feels) * 2.0
        score -= penalty
        reasons.append(f"{round(feels)}F feels cold")
    elif feels > IDEAL_TEMP_MAX:
        penalty = (feels - IDEAL_TEMP_MAX) * 2.5
        score -= penalty
        reasons.append(f"{round(feels)}F feels warm")

    precip = c["precipitation_probability"] or 0
    if precip > 20:
        score -= (precip - 20) * 1.2
        reasons.append(f"{round(precip)}% chance of rain")

    wind = c["wind_speed_10m"] or 0
    if wind > 12:
        score -= (wind - 12) * 1.5
        reasons.append(f"{round(wind)} mph wind")

    uv = c["uv_index"] or 0
    if uv > 5:
        score -= (uv - 5) * 6.0
        reasons.append(f"UV index {round(uv)}")

    return max(0.0, score), reasons


def pick_two(slots):
    """
    Best slot, then the best slot at least MIN_SEPARATION_MINUTES away.

    The separation rule is what keeps this from proposing 7:00 and 7:30 as two
    'options' -- they are the same walk.
    """
    scored = []
    for slot in slots:
        score, reasons = score_slot(slot)
        scored.append({**slot, "score": score, "reasons": reasons})

    scored.sort(key=lambda s: s["score"], reverse=True)
    if not scored:
        return []

    best = scored[0]
    gap = dt.timedelta(minutes=MIN_SEPARATION_MINUTES)
    runner_up = next(
        (s for s in scored[1:] if abs(s["start"] - best["start"]) >= gap), None
    )

    picks = [best] + ([runner_up] if runner_up else [])
    picks.sort(key=lambda s: s["start"])
    return picks


# --- Rationale -------------------------------------------------------------


def write_rationale(picks, day_label):
    """Ask Bedrock for a short, readable note. Falls back to a template."""
    summary = [
        {
            "time": p["start"].strftime("%-I:%M %p"),
            "score": round(p["score"]),
            "feels_like_f": round(p["conditions"]["apparent_temperature"]),
            "rain_chance_pct": round(p["conditions"]["precipitation_probability"] or 0),
            "wind_mph": round(p["conditions"]["wind_speed_10m"] or 0),
            "uv_index": round(p["conditions"]["uv_index"] or 0),
            "drawbacks": p["reasons"],
        }
        for p in picks
    ]

    prompt = (
        "You are writing a short note, sent the evening before, suggesting when "
        f"to take a 30-minute walk tomorrow ({day_label}). Here are the two "
        "windows already selected, with their conditions:\n\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Write two or three sentences total. Refer to the times as tomorrow's. "
        "Name both times, say what makes each one good, and mention any real "
        "drawback honestly. No preamble, no bullet points, no sign-off. "
        "Conversational."
    )

    try:
        resp = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.4},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        log.exception("Bedrock call failed; falling back to template")
        return _template_rationale(summary, day_label)


def _template_rationale(summary, day_label):
    lines = []
    for s in summary:
        line = f"{s['time']}: feels like {s['feels_like_f']}F, {s['rain_chance_pct']}% rain, {s['wind_mph']} mph wind."
        lines.append(line)
    return f"Two best walk windows for {day_label}:\n" + "\n".join(lines)


# --- Handler ---------------------------------------------------------------


def lambda_handler(event, context):
    forecast = fetch_forecast()
    target = target_day(forecast)
    day_label = target.strftime("%A")  # e.g. "Saturday"

    slots = build_slots(forecast, target)
    picks = pick_two(slots)

    if not picks:
        message = (
            f"No daylight hours inside your walking window on {day_label}. "
            "Nothing to suggest."
        )
    elif picks[0]["score"] < 40:
        # Everything is bad. Say so rather than dressing up the least-bad hour.
        message = (
            f"Nothing on {day_label} scores well -- rain, heat, or wind across "
            "the whole window. Best available is "
            f"{picks[0]['start'].strftime('%-I:%M %p')}, but it is a fair-weather-"
            "friend day at best."
        )
    else:
        message = write_rationale(picks, day_label)

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"Walk windows for {day_label}",
        Message=message,
    )

    log.info("Published %d picks for %s", len(picks), target.isoformat())
    return {
        "target_date": target.isoformat(),
        "picks": [
            {"start": p["start"].isoformat(), "score": round(p["score"])} for p in picks
        ],
        "message": message,
    }
