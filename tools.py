import json
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

_WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _fetch_open_meteo_json(url: str, parameters: dict[str, str]) -> dict[str, Any]:
    request_url = f"{url}?{urlencode(parameters)}"
    try:
        with urlopen(request_url, timeout=10) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Open-Meteo request failed: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Open-Meteo returned invalid JSON.") from error

    if not isinstance(data, dict):
        raise RuntimeError("Open-Meteo returned an unexpected response.")
    if data.get("error"):
        raise RuntimeError(data.get("reason", "Open-Meteo returned an error."))
    return data


def get_current_time() -> str:
    """Get the current time in the Asian/Baghdad timezone and return it as a formatted string."""
    baghdad_tz = ZoneInfo("Asia/Baghdad")
    current_time = datetime.now(baghdad_tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S")


def get_weather(location: str) -> str:
    """Get current weather for a city or other named location using Open-Meteo."""
    location = location.strip()
    if not location:
        raise ValueError("A city or location is required.")

    geocoding = _fetch_open_meteo_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": location, "count": "1", "language": "en", "format": "json"},
    )
    results = geocoding.get("results") or []
    if not results:
        raise ValueError(f"No location found for {location!r}.")

    place = results[0]
    if "latitude" not in place or "longitude" not in place:
        raise RuntimeError("Open-Meteo returned a location without coordinates.")

    forecast = _fetch_open_meteo_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": str(place["latitude"]),
            "longitude": str(place["longitude"]),
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        },
    )
    current = forecast.get("current")
    units = forecast.get("current_units")
    if not isinstance(current, dict) or not isinstance(units, dict):
        raise RuntimeError("Open-Meteo returned no current weather data.")

    fields = (
        "time",
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
    )
    missing_fields = [field for field in fields if field not in current]
    if missing_fields:
        raise RuntimeError(
            "Open-Meteo response is missing current weather field(s): "
            + ", ".join(missing_fields)
        )

    unit_fields = (
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
    )
    missing_units = [field for field in unit_fields if field not in units]
    if missing_units:
        raise RuntimeError(
            "Open-Meteo response is missing current weather unit(s): "
            + ", ".join(missing_units)
        )

    location_parts = [
        place.get("name"),
        place.get("admin1"),
        place.get("country"),
    ]
    location_label = ", ".join(dict.fromkeys(part for part in location_parts if part))
    weather_code = int(current["weather_code"])
    condition = _WEATHER_DESCRIPTIONS.get(weather_code, f"Weather code {weather_code}")

    return (
        f"Current weather for {location_label} at {current['time']} local time: "
        f"{condition}; temperature {current['temperature_2m']}"
        f"{units['temperature_2m']}, feels like "
        f"{current['apparent_temperature']}{units['apparent_temperature']}; "
        f"humidity {current['relative_humidity_2m']}%; precipitation "
        f"{current['precipitation']}{units['precipitation']}; wind "
        f"{current['wind_speed_10m']}{units['wind_speed_10m']}."
    )
