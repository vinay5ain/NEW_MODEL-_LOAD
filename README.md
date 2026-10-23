# FloodGuard AI

FloodGuard combines an Open-Meteo weather forecast with nearby Central Water
Commission (CWC) station rainfall observations to show an **indicative
rainfall-related risk**. It is not a validated flood prediction. Rainfall
thresholds are configurable examples, not official warning thresholds, and the
app does not calculate a flood probability.

## What you need

- Windows, macOS, or Linux
- Python 3.10 or newer
- An internet connection for Open-Meteo forecast data
- The supplied rainfall CSV files in the project `DATA` folder (nested folders
  are also searched)

No Open-Meteo API key is required. No AI service or API key is used.

## Run on Windows

Open PowerShell in the project folder:

```powershell
cd D:\path\to\FLoodGuardAI4th
```

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, run the commands without
activating it by using the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Start the app:

```powershell
python app.py
```

When the terminal reports that Flask is running, open
<http://127.0.0.1:5000> in your browser. Keep the terminal open while using the
app. Press **Ctrl+C** in that terminal to stop it.

## First use

1. Allow the browser to access your location, or enter latitude and longitude
   manually. Browser location is generally available on `localhost` or a
   secure HTTPS site.
2. FloodGuard requests current conditions and a seven-day forecast from
   Open-Meteo, including hourly precipitation for the next 24 hours.
3. On the first assessment, FloodGuard processes CSV files under `DATA` and
   matches the location to the nearest CWC station within the configured
   distance. Processing the supplied datasets can take a few minutes.
4. Review the risk status, station distance and observation time, rainfall
   indicators, and listed data limitations.

Generated files are saved in the git-ignored `processed` folder. You can also
run the data-processing step separately:

```powershell
python -c "from floodguard.data_processing import process_cwc_data; process_cwc_data()"
```

## Configure thresholds and station distance (optional)

The defaults work without configuration. To change them, copy the example
configuration:

```powershell
Copy-Item .env.example .env
```

Edit `.env`, then restart the app for changes to take effect:

- `FLOODGUARD_MAX_STATION_DISTANCE_KM` — maximum CWC station matching distance
  in kilometers (default `50`).
- `FLOODGUARD_FORECAST_3H_MM` — illustrative three-hour forecast rainfall
  threshold (default `30` mm).
- `FLOODGUARD_FORECAST_6H_MM` — illustrative six-hour forecast rainfall
  threshold (default `50` mm).
- `FLOODGUARD_FORECAST_24H_MM` — illustrative 24-hour forecast rainfall
  threshold (default `100` mm).
- `FLOODGUARD_OBSERVED_3H_MM` — illustrative observed three-hour rainfall
  threshold (default `30` mm).

These are demonstration thresholds, not official flood-warning values. `.env`
is excluded from Git.

## Data and limitations

The app searches recursively for CSV files under `DATA`. Each source needs
recognizable station, timestamp, and rainfall columns; common CWC column names
and normalized variants are supported. Keep original files unchanged.

Naive CWC timestamps are interpreted as India Standard Time and normalized to
UTC. The processor reports missing and invalid timestamps, rainfall, and
coordinates; excludes negative rainfall; preserves legitimate zero rainfall;
and deduplicates observations shared between files. When duplicates conflict,
the observation from the lexicographically first filename is retained and the
conflict is recorded.

Open-Meteo forecast values are gridded forecasts; CWC values are measurements
from individual stations. FloodGuard keeps these sources distinct. CWC
observations older than 72 hours are not treated as recent. If the matched
station is too distant or its data is stale, or the required forecast is
unavailable, the app reports insufficient data rather than implying local
conditions.

The supplied project contains rainfall datasets but no verified historical
flood-event labels. Therefore, FloodGuard does not train a flood classifier or
produce a numerical flood probability. Rainfall alone cannot confirm whether a
location will flood, and a lower rainfall-related risk does not guarantee
safety.

## Generated files

After processing, `processed` contains:

- `cwc_rainfall_features.csv` — cleaned observations and supported 1-, 3-, 6-,
  and 24-hour rolling rainfall features.
- `stations.csv` — station metadata and latest observations/features.
- `data_quality_report.json` — per-file data-quality counts and duplicate
  handling details.
- `model_readiness.json` — label availability and requirements for future
  flood-event model training.

If the app reports that no usable rainfall CSVs were found, put the source CSV
files in `DATA` and try the assessment again.

## API endpoints

- `GET /api/status` — service and processed-data status.
- `GET /api/quality` — generated data-quality report.
- `GET /api/model-readiness` — flood-event model readiness.
- `POST /api/stations` — nearby station matches for a JSON body containing
  `latitude` and `longitude`.
- `POST /api/assess` — station match and rainfall assessment. Provide
  `latitude`, `longitude`, and an `hourly` object containing `time`,
  `precipitation`, and optional `precipitation_probability` arrays.

## Run tests

From the project folder:

```powershell
python -m unittest discover -s tests -v
```
