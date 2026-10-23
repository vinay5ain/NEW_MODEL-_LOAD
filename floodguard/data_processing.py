from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"
PROCESSED_DIR = PROJECT_ROOT / "processed"
FEATURES_PATH = PROCESSED_DIR / "cwc_rainfall_features.csv"
STATIONS_PATH = PROCESSED_DIR / "stations.csv"
QUALITY_PATH = PROCESSED_DIR / "data_quality_report.json"
MODEL_READINESS_PATH = PROCESSED_DIR / "model_readiness.json"
CHUNK_SIZE = 100_000

ALIASES = {
    "station": {"station", "stationname"},
    "state": {"state", "statename"},
    "district": {"district", "districtname"},
    "latitude": {"latitude", "lat"},
    "longitude": {"longitude", "lon", "lng"},
    "timestamp": {"dataacquisitiontime", "timestamp", "datetime", "time"},
    "rainfall": {
        "telemetryhourlyrainfallmm",
        "hourlyrainfallmm",
        "rainfallmm",
        "rainfall",
        "precipitationmm",
    },
}

WINDOWS = {
    "rain_1h_mm": pd.Timedelta(hours=1),
    "rain_3h_mm": pd.Timedelta(hours=3),
    "rain_6h_mm": pd.Timedelta(hours=6),
    "rain_24h_mm": pd.Timedelta(hours=24),
}


class DataProcessingError(RuntimeError):
    """Raised when historical rainfall cannot be processed usefully."""


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _column_map(columns: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for column in columns:
        normalized = _normalized_column(column)
        for canonical, aliases in ALIASES.items():
            if normalized in aliases and canonical not in mapped:
                mapped[canonical] = column
    return mapped


def _valid_coordinates(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    return (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )


def _station_key(row: pd.Series) -> str:
    values = [
        str(row.get("station") or "").strip().casefold(),
        str(row.get("state") or "").strip().casefold(),
        str(row.get("district") or "").strip().casefold(),
        "" if pd.isna(row.get("latitude")) else f"{row['latitude']:.5f}",
        "" if pd.isna(row.get("longitude")) else f"{row['longitude']:.5f}",
    ]
    return "|".join(values)


def _parse_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            "Asia/Kolkata", ambiguous="NaT", nonexistent="NaT"
        )
    else:
        parsed = parsed.dt.tz_convert("Asia/Kolkata")
    return parsed.dt.tz_convert("UTC")


def _new_report() -> dict[str, Any]:
    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_directory": str(DATA_DIR),
        "timestamp_assumption": (
            "Naive CWC timestamps are interpreted as Asia/Kolkata, then normalized to UTC."
        ),
        "rainfall_unit": "mm per source observation; source columns are expected in mm.",
        "duplicate_resolution": (
            "For a station and timestamp, keep the record from the lexicographically "
            "first source filename; conflicting values are counted and sampled below."
        ),
        "files": [],
    }


def process_cwc_data(
    data_dir: Path = DATA_DIR, processed_dir: Path = PROCESSED_DIR
) -> dict[str, Any]:
    """Normalize CWC CSVs in chunks and create station-grouped rolling features."""
    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(data_dir.rglob("*.csv")) if data_dir.exists() else []
    report = _new_report()
    report["source_directory"] = str(data_dir)
    if not input_files:
        report["error"] = f"No CSV files were found recursively under {data_dir}."
        QUALITY_PATH_LOCAL = processed_dir / "data_quality_report.json"
        QUALITY_PATH_LOCAL.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise DataProcessingError(report["error"])

    features_path = processed_dir / "cwc_rainfall_features.csv"
    stations_path = processed_dir / "stations.csv"
    quality_path = processed_dir / "data_quality_report.json"
    readiness_path = processed_dir / "model_readiness.json"
    usable_rows = 0

    with tempfile.TemporaryDirectory(prefix="floodguard-") as temporary_directory:
        database_path = Path(temporary_directory) / "staging.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.execute(
            """
            CREATE TABLE measurements (
                station_key TEXT NOT NULL,
                station TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                timestamp TEXT NOT NULL,
                rainfall REAL NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX measurement_station_time ON measurements(station_key, timestamp)"
        )

        for path in input_files:
            file_report: dict[str, Any] = {
                "file": str(path.relative_to(data_dir)),
                "rows_read": 0,
                "usable_rows": 0,
                "missing_timestamp": 0,
                "invalid_timestamp": 0,
                "missing_rainfall": 0,
                "invalid_rainfall": 0,
                "negative_rainfall": 0,
                "invalid_coordinates": 0,
                "missing_station": 0,
            }
            try:
                header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
                columns = _column_map(list(header.columns))
                missing_required = [
                    name for name in ("station", "timestamp", "rainfall") if name not in columns
                ]
                if missing_required:
                    file_report["error"] = (
                        "Missing required columns: " + ", ".join(missing_required)
                    )
                    report["files"].append(file_report)
                    continue

                selected_columns = set(columns.values())
                for chunk in pd.read_csv(
                    path,
                    chunksize=CHUNK_SIZE,
                    dtype=str,
                    encoding="utf-8-sig",
                    usecols=lambda column: column in selected_columns,
                    on_bad_lines="error",
                ):
                    chunk = chunk.rename(
                        columns={source: canonical for canonical, source in columns.items()}
                    )
                    for optional in ("state", "district", "latitude", "longitude"):
                        if optional not in chunk:
                            chunk[optional] = ""
                    file_report["rows_read"] += len(chunk)

                    raw_rainfall = chunk["rainfall"].astype("string").str.strip()
                    rainfall = pd.to_numeric(raw_rainfall, errors="coerce")
                    file_report["missing_rainfall"] += int(
                        (raw_rainfall.isna() | raw_rainfall.eq("")).sum()
                    )
                    file_report["invalid_rainfall"] += int(
                        (raw_rainfall.notna() & raw_rainfall.ne("") & rainfall.isna()).sum()
                    )
                    file_report["negative_rainfall"] += int((rainfall < 0).sum())

                    raw_timestamp = chunk["timestamp"].astype("string").str.strip()
                    timestamp_missing = raw_timestamp.isna() | raw_timestamp.eq("")
                    timestamps = _parse_timestamp(raw_timestamp)
                    file_report["missing_timestamp"] += int(timestamp_missing.sum())
                    file_report["invalid_timestamp"] += int(
                        (~timestamp_missing & timestamps.isna()).sum()
                    )

                    chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
                    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
                    coordinates_valid = _valid_coordinates(chunk["latitude"], chunk["longitude"])
                    file_report["invalid_coordinates"] += int((~coordinates_valid).sum())
                    chunk["station"] = chunk["station"].fillna("").astype(str).str.strip()
                    file_report["missing_station"] += int(chunk["station"].eq("").sum())

                    valid = (
                        timestamps.notna()
                        & rainfall.notna()
                        & rainfall.ge(0)
                        & chunk["station"].ne("")
                    )
                    if not valid.any():
                        continue

                    normalized = chunk.loc[valid, [
                        "station", "state", "district", "latitude", "longitude"
                    ]].copy()
                    normalized["timestamp"] = timestamps.loc[valid].dt.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    normalized["rainfall"] = rainfall.loc[valid].astype(float)
                    normalized["source"] = path.name
                    normalized["station_key"] = normalized.apply(_station_key, axis=1)
                    normalized["state"] = normalized["state"].fillna("").astype(str).str.strip()
                    normalized["district"] = (
                        normalized["district"].fillna("").astype(str).str.strip()
                    )
                    normalized.loc[~coordinates_valid.loc[valid], ["latitude", "longitude"]] = np.nan
                    normalized = normalized[
                        [
                            "station_key", "station", "state", "district", "latitude",
                            "longitude", "timestamp", "rainfall", "source"
                        ]
                    ]
                    normalized.to_sql(
                        "measurements",
                        connection,
                        if_exists="append",
                        index=False,
                        chunksize=20_000,
                    )
                    count = int(len(normalized))
                    file_report["usable_rows"] += count
                    usable_rows += count
                connection.commit()
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
                file_report["error"] = f"{type(error).__name__}: {error}"
                connection.rollback()
            report["files"].append(file_report)

        if usable_rows == 0:
            report["error"] = (
                f"No usable CWC observations were found. Place CSV files under {data_dir} "
                "with station, timestamp, and rainfall columns."
            )
            quality_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            connection.close()
            raise DataProcessingError(report["error"])

        duplicate_summary = connection.execute(
            """
            SELECT
                COALESCE(SUM(group_count - 1), 0),
                COALESCE(SUM(CASE WHEN distinct_values > 1 THEN 1 ELSE 0 END), 0)
            FROM (
                SELECT station_key, timestamp, COUNT(*) AS group_count,
                       COUNT(DISTINCT rainfall) AS distinct_values
                FROM measurements
                GROUP BY station_key, timestamp
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
        conflicts = connection.execute(
            """
            SELECT station_key, timestamp, MIN(rainfall), MAX(rainfall), COUNT(*)
            FROM measurements
            GROUP BY station_key, timestamp
            HAVING COUNT(DISTINCT rainfall) > 1
            ORDER BY timestamp
            LIMIT 25
            """
        ).fetchall()
        report["duplicate_records_removed"] = int(duplicate_summary[0] or 0)
        report["conflicting_duplicate_groups"] = int(duplicate_summary[1] or 0)
        report["conflicting_duplicate_samples"] = [
            {
                "station_key": row[0],
                "timestamp_utc": row[1],
                "minimum_mm": row[2],
                "maximum_mm": row[3],
                "record_count": row[4],
            }
            for row in conflicts
        ]
        connection.execute(
            """
            CREATE TABLE resolved AS
            SELECT station_key, station, state, district, latitude, longitude, timestamp, rainfall
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY station_key, timestamp
                    ORDER BY source COLLATE BINARY, rowid
                ) AS row_number
                FROM measurements
            )
            WHERE row_number = 1
            """
        )
        connection.execute("CREATE INDEX resolved_station_time ON resolved(station_key, timestamp)")
        connection.commit()

        station_rows = connection.execute(
            """
            SELECT station_key, station, state, district, latitude, longitude,
                   COUNT(*) AS observation_count
            FROM resolved
            GROUP BY station_key
            ORDER BY station_key
            """
        ).fetchall()

        feature_columns = [
            "station_key", "station", "state", "district", "latitude", "longitude",
            "timestamp_utc", "rainfall_mm", "sampling_median_minutes",
            *WINDOWS.keys(),
        ]
        station_columns = [
            "station_key", "station", "state", "district", "latitude", "longitude",
            "observation_count", "last_observed_at_utc", "last_rainfall_mm",
            "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_24h_mm",
            "sampling_median_minutes",
        ]
        station_summaries: list[dict[str, Any]] = []
        wrote_features = False

        for station_row in station_rows:
            station_key = station_row[0]
            frame = pd.read_sql_query(
                """
                SELECT station_key, station, state, district, latitude, longitude,
                       timestamp, rainfall
                FROM resolved WHERE station_key = ? ORDER BY timestamp
                """,
                connection,
                params=(station_key,),
            )
            index = pd.to_datetime(frame.pop("timestamp"), utc=True)
            frame = frame.set_index(index)
            differences = frame.index.to_series().diff().dropna().dt.total_seconds() / 60
            sampling_median = float(differences.median()) if not differences.empty else None
            for feature, window in WINDOWS.items():
                supported = sampling_median is not None and sampling_median <= window.total_seconds() / 60
                if supported:
                    frame[feature] = frame["rainfall"].rolling(
                        window, min_periods=1, closed="both"
                    ).sum()
                else:
                    frame[feature] = np.nan
            frame["timestamp_utc"] = frame.index.strftime("%Y-%m-%dT%H:%M:%SZ")
            frame["rainfall_mm"] = frame.pop("rainfall")
            frame["sampling_median_minutes"] = sampling_median
            exported = frame.reset_index(drop=True)[feature_columns]
            exported.to_csv(
                features_path,
                mode="a" if wrote_features else "w",
                header=not wrote_features,
                index=False,
                na_rep="",
            )
            wrote_features = True

            last = frame.iloc[-1]
            station_summaries.append(
                {
                    "station_key": station_key,
                    "station": station_row[1],
                    "state": station_row[2],
                    "district": station_row[3],
                    "latitude": station_row[4],
                    "longitude": station_row[5],
                    "observation_count": int(station_row[6]),
                    "last_observed_at_utc": last["timestamp_utc"],
                    "last_rainfall_mm": float(last["rainfall_mm"]),
                    "rain_1h_mm": _as_finite_or_none(last["rain_1h_mm"]),
                    "rain_3h_mm": _as_finite_or_none(last["rain_3h_mm"]),
                    "rain_6h_mm": _as_finite_or_none(last["rain_6h_mm"]),
                    "rain_24h_mm": _as_finite_or_none(last["rain_24h_mm"]),
                    "sampling_median_minutes": sampling_median,
                }
            )

        pd.DataFrame(station_summaries, columns=station_columns).to_csv(
            stations_path, index=False, na_rep=""
        )
        connection.close()

    report["usable_observations_before_deduplication"] = usable_rows
    report["unique_station_timestamp_observations"] = usable_rows - report[
        "duplicate_records_removed"
    ]
    report["station_count"] = len(station_summaries)
    report["processed_files"] = {
        "features": str(features_path),
        "stations": str(stations_path),
    }
    readiness = {
        "verified_flood_event_labels_found": False,
        "validated_flood_prediction_model_trained": False,
        "status": "not_trainable_without_verified_labels",
        "reason": (
            "The supplied files contain CWC rainfall observations but no verified "
            "historical flood-event labels or verified non-flood examples."
        ),
        "required_data": [
            "Geocoded, timestamped flood-event records with a verifiable source.",
            "Event start/end times and location accuracy or affected-area geometry.",
            "Verified non-flood observation windows sampled under the same coverage criteria.",
            "Documentation of reporting gaps and event-verification methodology.",
        ],
    }
    report["model_readiness"] = readiness
    quality_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    return report


def _as_finite_or_none(value: Any) -> float | None:
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def processed_data_available(processed_dir: Path = PROCESSED_DIR) -> bool:
    processed_dir = Path(processed_dir)
    return all(
        (processed_dir / name).is_file()
        for name in (
            "cwc_rainfall_features.csv",
            "stations.csv",
            "data_quality_report.json",
            "model_readiness.json",
        )
    )
