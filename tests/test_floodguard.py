from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from floodguard.data_processing import DataProcessingError, process_cwc_data
from floodguard.risk_engine import assess_rainfall_risk, haversine_km, nearby_stations


class RiskEngineTests(unittest.TestCase):
    def test_haversine_and_station_distance_matching(self):
        distance = haversine_km(0, 0, 0, 1)
        self.assertAlmostEqual(distance, 111.195, places=2)
        result = nearby_stations(
            0,
            0,
            [
                {"station": "nearest", "latitude": 0, "longitude": 0.1},
                {"station": "invalid", "latitude": 200, "longitude": 0},
            ],
            max_distance_km=50,
        )
        self.assertEqual(result["matched_station"]["station"], "nearest")
        self.assertEqual(len(result["nearby_stations"]), 1)

    def test_assessment_never_invents_probability_and_requires_fresh_data(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        station = {
            "last_observed_at_utc": recent.isoformat(),
            "rain_3h_mm": 35,
        }
        forecast = {
            "forecast_3h_mm": 35,
            "forecast_6h_mm": 40,
            "forecast_24h_mm": 60,
            "valid_hours": 24,
        }
        result = assess_rainfall_risk(station, forecast)
        self.assertEqual(result["status"], "elevated_rainfall_risk")
        self.assertIsNone(result["flood_probability"])
        stale = assess_rainfall_risk(
            {**station, "last_observed_at_utc": "2020-01-01T00:00:00+00:00"},
            forecast,
        )
        self.assertEqual(stale["status"], "insufficient_data")


class DataProcessingTests(unittest.TestCase):
    header = (
        "Station,State,District,Latitude,Longitude,Data Acquisition Time,"
        "Telemetry Hourly Rainfall (mm)\n"
    )

    def test_chunk_processor_deduplicates_preserves_zero_and_reports_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "DATA"
            output = root / "processed"
            source.mkdir()
            (source / "a.csv").write_text(
                self.header
                + "Gauge,State,District,10,20,01-01-2024 00:00,0\n"
                + "Gauge,State,District,10,20,01-01-2024 00:30,5\n"
                + "Gauge,State,District,10,20,01-01-2024 01:00,0\n"
                + "Gauge,State,District,10,20,01-01-2024 01:30,-2\n"
                + "Gauge,State,District,10,20,not-a-date,1\n"
                + "Gauge,State,District,10,20,01-01-2024 02:00,\n",
                encoding="utf-8",
            )
            (source / "b.csv").write_text(
                self.header + "Gauge,State,District,10,20,01-01-2024 00:30,7\n",
                encoding="utf-8",
            )
            report = process_cwc_data(source, output)
            features = pd.read_csv(output / "cwc_rainfall_features.csv")
            self.assertEqual(len(features), 3)
            self.assertIn(0, features["rainfall_mm"].tolist())
            duplicate_row = features.loc[features["rainfall_mm"] == 5]
            self.assertEqual(float(duplicate_row.iloc[0]["rainfall_mm"]), 5.0)
            self.assertEqual(report["duplicate_records_removed"], 1)
            self.assertEqual(report["conflicting_duplicate_groups"], 1)
            self.assertEqual(report["station_count"], 1)
            self.assertEqual(report["files"][0]["negative_rainfall"], 1)
            self.assertEqual(report["files"][0]["invalid_timestamp"], 1)
            self.assertEqual(report["files"][0]["missing_rainfall"], 1)
            self.assertFalse(report["model_readiness"]["verified_flood_event_labels_found"])
            self.assertTrue((output / "data_quality_report.json").is_file())
            self.assertTrue((output / "model_readiness.json").is_file())
            self.assertEqual(
                json.loads((output / "data_quality_report.json").read_text())["station_count"],
                1,
            )

    def test_no_csv_produces_informative_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "DATA").mkdir()
            with self.assertRaisesRegex(DataProcessingError, "No CSV files were found"):
                process_cwc_data(root / "DATA", root / "processed")


if __name__ == "__main__":
    unittest.main()
