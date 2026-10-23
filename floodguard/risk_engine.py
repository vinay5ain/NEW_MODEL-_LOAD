from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    earth_radius_km = 6371.0088
    lat_delta = radians(latitude_b - latitude_a)
    lon_delta = radians(longitude_b - longitude_a)
    value = (
        sin(lat_delta / 2) ** 2
        + cos(radians(latitude_a))
        * cos(radians(latitude_b))
        * sin(lon_delta / 2) ** 2
    )
    return 2 * earth_radius_km * asin(sqrt(min(1.0, value)))


def nearby_stations(
    latitude: float,
    longitude: float,
    stations: list[dict[str, Any]],
    max_distance_km: float = 50.0,
    limit: int = 5,
) -> dict[str, Any]:
    candidates = []
    for station in stations:
        station_lat = _number(station.get("latitude"))
        station_lon = _number(station.get("longitude"))
        if station_lat is None or station_lon is None:
            continue
        if not (-90 <= station_lat <= 90 and -180 <= station_lon <= 180):
            continue
        distance = haversine_km(latitude, longitude, station_lat, station_lon)
        candidates.append({**station, "distance_km": round(distance, 2)})
    candidates.sort(key=lambda station: station["distance_km"])
    matched = candidates[0] if candidates and candidates[0]["distance_km"] <= max_distance_km else None
    return {
        "matched_station": matched,
        "nearby_stations": candidates[:limit],
        "max_matching_distance_km": max_distance_km,
    }


def assess_rainfall_risk(
    station: dict[str, Any] | None,
    forecast: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    max_station_age_hours: float = 72,
) -> dict[str, Any]:
    """Produce a transparent rainfall indicator, never a flood probability."""
    thresholds = thresholds or {
        "forecast_3h_mm": 30.0,
        "forecast_6h_mm": 50.0,
        "forecast_24h_mm": 100.0,
        "observed_3h_mm": 30.0,
    }
    now = datetime.now(timezone.utc)
    station_age_hours = None
    station_is_fresh = False
    if station and station.get("last_observed_at_utc"):
        try:
            observed_at = datetime.fromisoformat(
                str(station["last_observed_at_utc"]).replace("Z", "+00:00")
            )
            station_age_hours = max(0.0, (now - observed_at).total_seconds() / 3600)
            station_is_fresh = station_age_hours <= max_station_age_hours
        except ValueError:
            station_age_hours = None

    indicators: list[dict[str, Any]] = []
    for key, threshold_key, label in (
        ("forecast_3h_mm", "forecast_3h_mm", "Forecast rainfall over 3 hours"),
        ("forecast_6h_mm", "forecast_6h_mm", "Forecast rainfall over 6 hours"),
        ("forecast_24h_mm", "forecast_24h_mm", "Forecast rainfall over 24 hours"),
    ):
        value = _number(forecast.get(key))
        if value is not None:
            indicators.append(
                {
                    "name": label,
                    "value_mm": round(value, 2),
                    "threshold_mm": thresholds[threshold_key],
                    "exceeds_configured_threshold": value >= thresholds[threshold_key],
                    "source": "Open-Meteo gridded forecast",
                }
            )
    probability = _number(forecast.get("precipitation_probability_max"))
    if probability is not None:
        indicators.append(
            {
                "name": "Maximum forecast precipitation probability in the next 24 hours",
                "value_percent": round(min(probability, 100.0), 1),
                "exceeds_configured_threshold": False,
                "source": "Open-Meteo gridded forecast",
            }
        )
    observed_3h = _number(station.get("rain_3h_mm")) if station else None
    if station_is_fresh and observed_3h is not None:
        indicators.append(
            {
                "name": "Observed rainfall over the station's last 3-hour window",
                "value_mm": round(observed_3h, 2),
                "threshold_mm": thresholds["observed_3h_mm"],
                "exceeds_configured_threshold": observed_3h >= thresholds["observed_3h_mm"],
                "source": "CWC station observation",
            }
        )

    exceeds_threshold = any(item["exceeds_configured_threshold"] for item in indicators)
    forecast_coverage = int(forecast.get("valid_hours", 0))
    essential_data_available = (
        station is not None
        and station_is_fresh
        and forecast_coverage >= 24
        and forecast.get("forecast_24h_mm") is not None
    )
    if not essential_data_available:
        status = "insufficient_data"
        label = "Insufficient data"
    elif exceeds_threshold:
        status = "elevated_rainfall_risk"
        label = "Elevated rainfall-related risk"
    else:
        status = "lower_rainfall_risk"
        label = "Lower rainfall-related risk"

    limitations = [
        "This is an indicative rainfall-based assessment, not a validated flood prediction.",
        "Configured thresholds are illustrative and are not official flood-warning thresholds.",
        "Rainfall alone does not account for drainage, terrain, river levels, or local flood reports.",
        "A lower rainfall-related indicator does not guarantee safety.",
    ]
    if not station:
        limitations.append("No CWC station was found within the configured matching distance.")
    elif not station_is_fresh:
        limitations.append(
            "The matched CWC station observation is missing, invalid, or older than "
            f"{max_station_age_hours:g} hours; it is not treated as recent rainfall."
        )
    if forecast_coverage < 24:
        limitations.append(
            "A complete 24-hour forecast rainfall window is unavailable."
        )

    return {
        "status": status,
        "label": label,
        "indicators": indicators,
        "forecast_coverage_hours": forecast_coverage,
        "station_observation_age_hours": (
            round(station_age_hours, 1) if station_age_hours is not None else None
        ),
        "verified_historical_flood_evidence": False,
        "flood_probability": None,
        "limitations": limitations,
        "thresholds": thresholds,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 and number < float("inf") else None
