from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from floodguard.data_processing import (
    DATA_DIR,
    MODEL_READINESS_PATH,
    PROCESSED_DIR,
    QUALITY_PATH,
    STATIONS_PATH,
    DataProcessingError,
    process_cwc_data,
    processed_data_available,
)
from floodguard.risk_engine import assess_rainfall_risk, nearby_stations

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
PROCESS_LOCK = threading.Lock()


def _ensure_processed() -> None:
    if processed_data_available(PROCESSED_DIR):
        return
    with PROCESS_LOCK:
        if not processed_data_available(PROCESSED_DIR):
            process_cwc_data(DATA_DIR, PROCESSED_DIR)


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _coordinates(body: dict[str, Any]) -> tuple[float, float]:
    latitude = _json_number(body.get("latitude"))
    longitude = _json_number(body.get("longitude"))
    if (
        latitude is None
        or longitude is None
        or not (-90 <= latitude <= 90)
        or not (-180 <= longitude <= 180)
    ):
        raise ValueError("Latitude must be between -90 and 90 and longitude between -180 and 180.")
    return latitude, longitude


def _load_stations() -> list[dict[str, Any]]:
    if not STATIONS_PATH.is_file():
        return []
    return pd.read_csv(STATIONS_PATH).replace({np.nan: None}).to_dict(orient="records")


def _forecast_summary(body: dict[str, Any]) -> dict[str, Any]:
    hourly = body.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("A forecast hourly object is required.")
    times = hourly.get("time")
    precipitation = hourly.get("precipitation")
    probability = hourly.get("precipitation_probability", [])
    if (
        not isinstance(times, list)
        or not isinstance(precipitation, list)
        or (probability is not None and not isinstance(probability, list))
        or len(times) != len(precipitation)
        or len(times) > 200
        or (probability and len(probability) != len(times))
    ):
        raise ValueError("Forecast hourly time, precipitation, and probability arrays are malformed.")
    observations: list[tuple[str, float | None, float | None]] = []
    for index, (timestamp, amount) in enumerate(zip(times, precipitation)):
        amount_number = _json_number(amount)
        chance = _json_number(probability[index]) if probability else None
        if chance is not None and not 0 <= chance <= 100:
            chance = None
        if isinstance(timestamp, str):
            observations.append(
                (
                    timestamp,
                    amount_number if amount_number is not None and amount_number >= 0 else None,
                    chance,
                )
            )
    observations.sort(key=lambda item: item[0])
    selected = observations[:24]
    valid_hours = 0
    previous_time = None
    for timestamp, amount, _ in selected:
        try:
            parsed_time = pd.Timestamp(timestamp)
        except (TypeError, ValueError):
            break
        if (
            amount is None
            or (
                previous_time is not None
                and parsed_time - previous_time != pd.Timedelta(hours=1)
            )
        ):
            break
        valid_hours += 1
        previous_time = parsed_time
    totals: dict[str, float | None] = {}
    for hours in (3, 6, 24):
        items = selected[:hours]
        totals[f"forecast_{hours}h_mm"] = (
            round(sum(item[1] for item in items), 2) if valid_hours >= hours else None
        )
    return {
        **totals,
        "valid_hours": valid_hours,
        "precipitation_probability_max": (
            max((item[2] for item in selected[:valid_hours] if item[2] is not None), default=None)
        ),
    }


@app.get("/")
def index():
    return send_from_directory(PROJECT_ROOT, "floodguard_current_location_7day.html")


@app.get("/api/status")
def status():
    return jsonify(
        {
            "status": "ok",
            "data_available": processed_data_available(PROCESSED_DIR),
            "source_directory": str(DATA_DIR),
            "risk_mode": "indicative_rainfall_only",
        }
    )


@app.get("/api/quality")
def quality():
    if not QUALITY_PATH.is_file():
        return jsonify({"available": False, "message": "Historical data has not been processed yet."})
    return jsonify(json.loads(QUALITY_PATH.read_text(encoding="utf-8")))


@app.get("/api/model-readiness")
def model_readiness():
    if not MODEL_READINESS_PATH.is_file():
        return jsonify(
            {
                "verified_flood_event_labels_found": False,
                "validated_flood_prediction_model_trained": False,
                "status": "not_trainable_without_verified_labels",
                "reason": "No verified flood-event labels are included in the supplied project files.",
            }
        )
    return jsonify(json.loads(MODEL_READINESS_PATH.read_text(encoding="utf-8")))


@app.post("/api/stations")
def stations():
    try:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        latitude, longitude = _coordinates(body)
        _ensure_processed()
        max_distance = float(os.environ.get("FLOODGUARD_MAX_STATION_DISTANCE_KM", "50"))
        if not np.isfinite(max_distance) or not 0 < max_distance <= 500:
            raise ValueError("FLOODGUARD_MAX_STATION_DISTANCE_KM must be in (0, 500].")
        return jsonify(
            nearby_stations(
                latitude, longitude, _load_stations(), max_distance_km=max_distance
            )
        )
    except DataProcessingError as error:
        return jsonify({"error": str(error)}), 503
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except OSError as error:
        app.logger.exception("Unable to read processed rainfall data")
        return jsonify({"error": f"Unable to read processed rainfall data: {error}"}), 500


@app.post("/api/assess")
def assess():
    try:
        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        latitude, longitude = _coordinates(body)
        forecast = _forecast_summary(body)
        _ensure_processed()
        max_distance = float(os.environ.get("FLOODGUARD_MAX_STATION_DISTANCE_KM", "50"))
        if not np.isfinite(max_distance) or not 0 < max_distance <= 500:
            raise ValueError("FLOODGUARD_MAX_STATION_DISTANCE_KM must be in (0, 500].")
        match = nearby_stations(
            latitude, longitude, _load_stations(), max_distance_km=max_distance
        )
        station = match["matched_station"]
        thresholds = {
            "forecast_3h_mm": _threshold("FLOODGUARD_FORECAST_3H_MM", 30),
            "forecast_6h_mm": _threshold("FLOODGUARD_FORECAST_6H_MM", 50),
            "forecast_24h_mm": _threshold("FLOODGUARD_FORECAST_24H_MM", 100),
            "observed_3h_mm": _threshold("FLOODGUARD_OBSERVED_3H_MM", 30),
        }
        risk = assess_rainfall_risk(station, forecast, thresholds)
        return jsonify(
            {
                "location": {"latitude": latitude, "longitude": longitude},
                "station_match": match,
                "forecast": forecast,
                "risk": risk,
                "data_sources": {
                    "historical": "CWC station observations",
                    "forecast": "Open-Meteo gridded forecast",
                    "verified_flood_event_labels_found": False,
                },
            }
        )
    except DataProcessingError as error:
        return jsonify({"error": str(error)}), 503
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except OSError as error:
        app.logger.exception("Unable to assess rainfall risk")
        return jsonify({"error": f"Unable to assess rainfall risk: {error}"}), 500


def _threshold(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number.") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return value
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
