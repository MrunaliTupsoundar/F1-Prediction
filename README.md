# F1 Predictor — Baku

Compact research prototype that uses FastF1 + a small ML pipeline to produce illustrative race-time predictions for the Azerbaijan Grand Prix.

---

## Features

- Downloads session data via fastf1 (cache enabled by default).
- Builds per-driver features (sector means, qualifying, team score, weather).
- Trains a Gradient Boosting model and prints predicted race times.
- Produces a scatter plot and a ranked table of predictions.
- Simple, dependency-free `.env` loader for configuration.

---

## Tech Stack

- Python (3.10+ recommended)
- fastf1, pandas, numpy, scikit-learn, matplotlib, requests

---

## Project Structure

```
d:\f1-predictor/
│
├── README.md
├── .gitignore
├── .env                # template (do NOT commit real keys)
├── requirements.txt
├── baku.py             # main script (loads data, trains model, plots)
├── main.py             # alternate entry (if present)
└── f1_cache/           # fastf1 cache (created at runtime; ignored by git)
```

---

## Installation & Setup (Cross-platform)

1. Clone or open the repository root.

2. Create & activate a virtual environment (choose the section for your OS).

Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (cmd)
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Linux / macOS (bash / zsh)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If your platform uses `python` for Python 3, replace `python3` with `python`.
- To deactivate the venv run: `deactivate`.

3. Ensure cache directory exists

Windows (PowerShell / cmd)
```powershell
mkdir f1_cache
```

Linux / macOS
```bash
mkdir -p f1_cache
```

4. Provide configuration in a `.env` file (project root) or set env vars in your shell.

---

## Environment variables / .env

Supported environment variables (all optional except WEATHERAPI_KEY for real forecasts):

- WEATHERAPI_KEY — your WeatherAPI key (no surrounding quotes)
- CACHE_DIR — cache path for fastf1 (default: `f1_cache`)
- RACE_DATE — forecast date (YYYY-MM-DD)
- RACE_HOUR — forecast hour (0-23)

Example `.env` (template included in repo):
```
WEATHERAPI_KEY=your_weatherapi_key_here
CACHE_DIR=f1_cache
RACE_DATE=2025-11-03
RACE_HOUR=15
```

You can also set the key in the current session:

PowerShell
```powershell
$env:WEATHERAPI_KEY = 'your_real_key'
```

bash/zsh
```bash
export WEATHERAPI_KEY='your_real_key'
```

Note: `.env` is excluded from Git via `.gitignore`. Do not commit secrets.

---

## Run

From the activated virtual environment:
```bash
python .\baku.py     # Windows PowerShell/cmd
python ./baku.py     # Linux/macOS
```

What to expect:
- The script ensures the cache directory exists (honors `CACHE_DIR`).
- Downloads relevant FastF1 session data (cached under `f1_cache/`).
- Builds features (some values in source are placeholders).
- Trains a Gradient Boosting model and prints predicted race times.
- Shows a scatter plot.

---

## Caveats & Next Steps

- Toy/demo project: training data is small (single race). Update placeholders (clean-air pace, qualifying inputs) for better results.
- Optional improvements:
  - Use `python-dotenv` to parse `.env` (if preferred).
  - Add unit/dry-run tests to validate script without contacting WeatherAPI.
  - Replace placeholders with real qualifying/pace data.

---

## Notes on Git & Secrets

- Ensure `.env`, `venv/`, `node_modules/`, and `f1_cache/` are in `.gitignore` to avoid committing secrets and large artifacts.

---

## License

MIT
```// filepath: d:\f1-predictor\README.md
# F1 Predictor — Baku

Compact research prototype that uses FastF1 + a small ML pipeline to produce illustrative race-time predictions for the Azerbaijan Grand Prix.

---

## Features

- Downloads session data via fastf1 (cache enabled by default).
- Builds per-driver features (sector means, qualifying, team score, weather).
- Trains a Gradient Boosting model and prints predicted race times.
- Produces a scatter plot and a ranked table of predictions.
- Simple, dependency-free `.env` loader for configuration.

---

## Tech Stack

- Python (3.10+ recommended)
- fastf1, pandas, numpy, scikit-learn, matplotlib, requests

---

## Project Structure

```
d:\f1-predictor/
│
├── README.md
├── .gitignore
├── .env                # template (do NOT commit real keys)
├── requirements.txt
├── baku.py             # main script (loads data, trains model, plots)
├── main.py             # alternate entry (if present)
└── f1_cache/           # fastf1 cache (created at runtime; ignored by git)
```

---

## Installation & Setup (Cross-platform)

1. Clone or open the repository root.

2. Create & activate a virtual environment (choose the section for your OS).

Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (cmd)
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Linux / macOS (bash / zsh)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If your platform uses `python` for Python 3, replace `python3` with `python`.
- To deactivate the venv run: `deactivate`.

3. Ensure cache directory exists

Windows (PowerShell / cmd)
```powershell
mkdir f1_cache
```

Linux / macOS
```bash
mkdir -p f1_cache
```

4. Provide configuration in a `.env` file (project root) or set env vars in your shell.

---

## Environment variables / .env

Supported environment variables (all optional except WEATHERAPI_KEY for real forecasts):

- WEATHERAPI_KEY — your WeatherAPI key (no surrounding quotes)
- CACHE_DIR — cache path for fastf1 (default: `f1_cache`)
- RACE_DATE — forecast date (YYYY-MM-DD)
- RACE_HOUR — forecast hour (0-23)

Example `.env` (template included in repo):
```
WEATHERAPI_KEY=your_weatherapi_key_here
CACHE_DIR=f1_cache
RACE_DATE=2025-11-03
RACE_HOUR=15
```

You can also set the key in the current session:

PowerShell
```powershell
$env:WEATHERAPI_KEY = 'your_real_key'
```

bash/zsh
```bash
export WEATHERAPI_KEY='your_real_key'
```

Note: `.env` is excluded from Git via `.gitignore`. Do not commit secrets.

---

## Run

From the activated virtual environment:
```bash
python .\baku.py     # Windows PowerShell/cmd
python ./baku.py     # Linux/macOS
```

What to expect:
- The script ensures the cache directory exists (honors `CACHE_DIR`).
- Downloads relevant FastF1 session data (cached under `f1_cache/`).
- Builds features (some values in source are placeholders).
- Trains a Gradient Boosting model and prints predicted race times.
- Shows a scatter plot.

---

## Caveats & Next Steps

- Toy/demo project: training data is small (single race). Update placeholders (clean-air pace, qualifying inputs) for better results.
- Optional improvements:
  - Use `python-dotenv` to parse `.env` (if preferred).
  - Add unit/dry-run tests to validate script without contacting WeatherAPI.
  - Replace placeholders with real qualifying/pace data.

---

## Notes on Git & Secrets

- Ensure `.env`, `venv/`, `node_modules/`, and `f1_cache/` are in `.gitignore` to avoid committing secrets and large artifacts.

---

## License

MIT