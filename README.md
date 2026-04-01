# F1 Predictor

FastF1-based race prediction prototype that trains on recent races, applies event-based backtesting, and predicts the next race using target qualifying and weather inputs.

## What It Does

- Collects race and qualifying data from recent Grand Prix weekends.
- Builds per-driver features (qualifying time/rank, sector pace proxy, team strength, weather).
- Trains a Gradient Boosting regression model on historical driver-event rows.
- Evaluates with event-based holdout backtest (not random row split).
- Predicts and ranks drivers for the configured target race.
- Plots clean-air pace versus predicted race time.

## Requirements

- Python 3.10+
- Dependencies in requirements.txt

Install:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration (.env)

The script reads env vars from .env if present.

### Core settings

- WEATHERAPI_KEY: WeatherAPI key for live weather. If empty/missing, defaults are used.
- TARGET_YEAR: Race year to predict (for example 2026).
- TARGET_EVENT: Event name (for example Japan).
- TRAIN_RACES: Number of recent completed races used for training (for example 10).
- CACHE_DIR: FastF1 cache directory (default f1_cache).

### Weather settings

- TRACK_LAT_LON: Track latitude/longitude (default Suzuka).
- AUTO_WEATHER_DATE: true/false. If true, uses today's date each run.
- RACE_DATE: Used only when AUTO_WEATHER_DATE=false.
- RACE_HOUR: Forecast hour (0-23).

Example .env:

```dotenv
WEATHERAPI_KEY=your_weatherapi_key_here
TARGET_YEAR=2026
TARGET_EVENT=Japan
TRAIN_RACES=10
CACHE_DIR=f1_cache
TRACK_LAT_LON=34.8431,136.5410
AUTO_WEATHER_DATE=true
RACE_HOUR=15
```

## Run

```powershell
python baku.py
```

Expected output:

- Training summary (races and rows used).
- Weather source/result message.
- Ranked prediction table for target event.
- Backtest MAE (event split).
- Scatter plot window.

## Notes on Model Realism

- Backtest MAE now uses event holdout and is more realistic than random row split.
- Rookie drivers may still use fallback feature values when history is sparse.
- This remains a lightweight predictor and not a full race-strategy simulation.

## Git Safety

- Do not commit secrets in .env.
- Keep .env, venv, and f1_cache ignored in .gitignore.

## License

MIT