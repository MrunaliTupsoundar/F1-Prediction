import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import os 
from datetime import datetime

# --- DOTENV LOADER & SETUP ---
def _load_dotenv(path=".env"):
    """Lightweight .env loader: reads KEY=VALUE lines and sets them in os.environ
    if not already present. Ignores comments and blank lines. Does not require
    external dependencies so this works out-of-the-box.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # remove surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # don't overwrite existing environment values
                os.environ.setdefault(key, val)
    except Exception:
        # Keep the loader robust: if anything fails, fall back to defaults.
        pass


# Load .env if present (project root .env)
_load_dotenv()

# Define the cache directory path (can be overridden by .env CACHE_DIR)
cache_dir = os.environ.get("CACHE_DIR", "f1_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# --- CONFIGURATION (DEFAULT: JAPANESE GP 2026) ---
TARGET_YEAR = int(os.environ.get("TARGET_YEAR", "2026"))
TARGET_EVENT = os.environ.get("TARGET_EVENT", "Japan")
TRAIN_RACES = int(os.environ.get("TRAIN_RACES", "24"))
TRACK_LAT_LON = os.environ.get("TRACK_LAT_LON", "34.8431,136.5410")
RACE_DATE = os.environ.get("RACE_DATE")
AUTO_WEATHER_DATE = os.environ.get("AUTO_WEATHER_DATE", "true").strip().lower() in {"1", "true", "yes", "on"}
try:
    RACE_HOUR = int(os.environ.get("RACE_HOUR", "14"))
except ValueError:
    RACE_HOUR = 14


def resolve_weather_date() -> str:
    """Return a weather date for API queries.

    If AUTO_WEATHER_DATE is enabled, today's local date is used on each run.
    Otherwise, RACE_DATE is respected, with today's date as a safe fallback.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if AUTO_WEATHER_DATE:
        return today
    if RACE_DATE:
        return RACE_DATE
    return today


def load_session_or_fail(year: int, event_name: str, session_type: str):
    """Load a FastF1 session and provide a concise error if unavailable."""
    try:
        session = fastf1.get_session(year, event_name, session_type)
        session.load()
        return session
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {year} {event_name} {session_type}. "
            f"Check internet/cache and event naming. Original error: {exc}"
        ) from exc


def load_session_or_none(year: int, event_identifier, session_type: str):
    """Best-effort session loader for multi-race training collection."""
    try:
        session = fastf1.get_session(year, event_identifier, session_type)
        session.load()
        return session
    except Exception:
        return None


def qualifying_times_from_session(session) -> pd.DataFrame:
    """Extract each driver's best qualifying lap time in seconds."""
    q_laps = session.laps[["Driver", "LapTime"]].dropna().copy()
    q_laps["QualifyingTime (s)"] = q_laps["LapTime"].dt.total_seconds()
    return (
        q_laps.groupby("Driver", as_index=False)["QualifyingTime (s)"]
        .min()
    )


def clean_air_pace_from_race(session) -> pd.Series:
    """Proxy for race pace: median of a driver's quick laps from race session."""
    race_laps = session.laps.pick_quicklaps(1.07)[["Driver", "LapTime"]].dropna().copy()
    race_laps["LapTime (s)"] = race_laps["LapTime"].dt.total_seconds()
    return race_laps.groupby("Driver")["LapTime (s)"].median()


def team_scores_from_race_results(session) -> pd.Series:
    """Create normalized team score from race finish points."""
    results = session.results[["TeamName", "Points"]].copy()
    team_points = results.groupby("TeamName")["Points"].sum()
    max_points = team_points.max() if not team_points.empty else 1
    return (team_points / max_points).rename("TeamPerformanceScore")


def driver_team_map_from_results(session) -> pd.Series:
    """Build a driver abbreviation -> team mapping from session results."""
    if session is None or getattr(session, "results", None) is None:
        return pd.Series(dtype="object")
    results = session.results.copy()
    if "Abbreviation" not in results.columns:
        return pd.Series(dtype="object")

    team_col = "TeamName" if "TeamName" in results.columns else None
    if team_col is None and "Team" in results.columns:
        team_col = "Team"
    if team_col is None:
        return pd.Series(dtype="object")

    mapping = results[["Abbreviation", team_col]].dropna()
    return mapping.drop_duplicates(subset=["Abbreviation"]).set_index("Abbreviation")[team_col]


def event_row_mask(schedule: pd.DataFrame, event_name: str) -> pd.Series:
    """Find matching event rows across common schedule columns."""
    name = event_name.strip().lower()
    mask = pd.Series(False, index=schedule.index)
    for col in ["EventName", "OfficialEventName", "Country", "Location"]:
        if col in schedule.columns:
            text = schedule[col].astype(str).str.lower()
            mask = mask | text.str.contains(name, regex=False)
    return mask


def get_recent_race_identifiers(target_year: int, target_event: str, n_races: int):
    """Return up to n_races completed rounds before target event, newest first."""
    target_round = None
    try:
        sched_target = fastf1.get_event_schedule(target_year, include_testing=False)
        mask = event_row_mask(sched_target, target_event)
        if mask.any() and "RoundNumber" in sched_target.columns:
            target_round = int(sched_target.loc[mask, "RoundNumber"].iloc[0])
    except Exception:
        pass

    recent = []
    for year in range(target_year, max(2018, target_year - 8), -1):
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception:
            continue
        if "RoundNumber" not in sched.columns:
            continue

        rounds = sorted(sched["RoundNumber"].dropna().astype(int).unique(), reverse=True)
        for rnd in rounds:
            if year == target_year and target_round is not None and rnd >= target_round:
                continue
            recent.append((year, int(rnd)))
            if len(recent) >= n_races:
                return recent

    return recent


def weather_from_race_session(session):
    """Extract approximate weather features from FastF1 race session weather data."""
    default_temp = 20.0
    default_rain = 0.0
    weather = getattr(session, "weather_data", None)
    if weather is None or weather.empty:
        return default_rain, default_temp

    temp = float(weather["AirTemp"].dropna().mean()) if "AirTemp" in weather.columns else default_temp
    if "Rainfall" in weather.columns:
        rain = weather["Rainfall"].dropna().astype(float)
        rain_prob = float(rain.mean()) if not rain.empty else default_rain
    else:
        rain_prob = default_rain
    return rain_prob, temp


def build_training_rows(race_session, quali_session, event_year: int, event_round: int) -> pd.DataFrame:
    """Build per-driver training rows for one event."""
    laps = race_session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna().copy()
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()

    sector = laps.groupby("Driver", as_index=False).agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    })
    sector["TotalSectorTime (s)"] = (
        sector["Sector1Time (s)"]
        + sector["Sector2Time (s)"]
        + sector["Sector3Time (s)"]
    )

    qualifying = qualifying_times_from_session(quali_session)
    clean_air = clean_air_pace_from_race(race_session).rename("CleanAirRacePace (s)")
    team_scores = team_scores_from_race_results(race_session)
    driver_team = driver_team_map_from_results(race_session)
    rain_probability, temperature = weather_from_race_session(race_session)

    y_map = laps.groupby("Driver")["LapTime (s)"].mean().rename("TargetLapTime (s)")

    rows = qualifying.copy()
    rows["CleanAirRacePace (s)"] = rows["Driver"].map(clean_air)
    rows["Team"] = rows["Driver"].map(driver_team)
    rows["TeamPerformanceScore"] = rows["Team"].map(team_scores)
    rows = rows.merge(sector[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    rows["RainProbability"] = rain_probability
    rows["Temperature"] = temperature
    rows["QualifyingRank"] = rows["QualifyingTime (s)"].rank(method="min")
    rows["QualifyingTime"] = rows["QualifyingTime (s)"] ** 2
    rows["TargetLapTime (s)"] = rows["Driver"].map(y_map)
    rows["EventYear"] = event_year
    rows["EventRound"] = event_round
    return rows


# --- STEP 1: BUILD TRAINING SET FROM MOST RECENT RACES ---
race_identifiers = get_recent_race_identifiers(TARGET_YEAR, TARGET_EVENT, TRAIN_RACES)
if not race_identifiers:
    raise RuntimeError("No recent races found for training. Check schedule availability.")

training_frames = []
recent_reference_race = None

for year, rnd in race_identifiers:
    race_session = load_session_or_none(year, rnd, "R")
    quali_session = load_session_or_none(year, rnd, "Q")
    if race_session is None or quali_session is None:
        continue
    try:
        event_rows = build_training_rows(race_session, quali_session, year, rnd)
    except Exception:
        continue
    if event_rows.empty:
        continue
    training_frames.append(event_rows)
    if recent_reference_race is None:
        recent_reference_race = race_session

if not training_frames:
    raise RuntimeError("Could not build training data from recent races. Check cache/internet.")

training_data = pd.concat(training_frames, ignore_index=True)
print(f"Training with {len(training_frames)} races and {len(training_data)} driver-event rows.")

if recent_reference_race is None:
    raise RuntimeError("No valid reference race available for target feature defaults.")

ref_clean_air = clean_air_pace_from_race(recent_reference_race)
ref_driver_team = driver_team_map_from_results(recent_reference_race)
ref_team_scores = team_scores_from_race_results(recent_reference_race)

ref_laps = recent_reference_race.laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna().copy()
for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
    ref_laps[f"{col} (s)"] = ref_laps[col].dt.total_seconds()
ref_sector = ref_laps.groupby("Driver", as_index=False).agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
})
ref_sector["TotalSectorTime (s)"] = (
    ref_sector["Sector1Time (s)"]
    + ref_sector["Sector2Time (s)"]
    + ref_sector["Sector3Time (s)"]
)
ref_sector_map = ref_sector.set_index("Driver")["TotalSectorTime (s)"]

qualifying_train_fallback = (
    training_data.groupby("Driver", as_index=False)["QualifyingTime (s)"]
    .median()
)
driver_to_team_train = (
    training_data[["Driver", "Team"]]
    .dropna()
    .drop_duplicates(subset=["Driver"], keep="first")
    .set_index("Driver")["Team"]
)

driver_hist_features = (
    training_data.groupby("Driver")[
        ["CleanAirRacePace (s)", "TotalSectorTime (s)", "TeamPerformanceScore"]
    ]
    .median()
)

team_hist_features = (
    training_data.dropna(subset=["Team"]).groupby("Team")[
        ["CleanAirRacePace (s)", "TotalSectorTime (s)", "TeamPerformanceScore"]
    ]
    .median()
)

# --- STEP 2: LOAD TARGET QUALIFYING (2026 JAPANESE GP) ---
quali_target = None
try:
    quali_target = load_session_or_fail(TARGET_YEAR, TARGET_EVENT, "Q")
    qualifying_target = qualifying_times_from_session(quali_target)
except Exception:
    print(
        f"WARNING: Could not load qualifying for {TARGET_YEAR} {TARGET_EVENT}. "
        "Using recent historical median qualifying as fallback."
    )
    qualifying_target = qualifying_train_fallback.copy()

qualifying_target["CleanAirRacePace (s)"] = qualifying_target["Driver"].map(ref_clean_air)
driver_to_team_target = driver_team_map_from_results(quali_target)
qualifying_target["Team"] = qualifying_target["Driver"].map(driver_to_team_target)
qualifying_target["Team"] = qualifying_target["Team"].fillna(
    qualifying_target["Driver"].map(ref_driver_team)
)
qualifying_target["Team"] = qualifying_target["Team"].fillna(
    qualifying_target["Driver"].map(driver_to_team_train)
)
qualifying_target["TeamPerformanceScore"] = qualifying_target["Team"].map(ref_team_scores)
qualifying_target["QualifyingRank"] = qualifying_target["QualifyingTime (s)"].rank(method="min")

# --- STEP 3: WEATHER DATA (USING WEATHERAPI.COM) ---

# Read Weather API key from environment (or .env loaded above)
API_KEY = os.environ.get("WEATHERAPI_KEY") or os.environ.get("API_KEY") or ""
if API_KEY:
    # sanitize common cases where the value might include surrounding quotes or whitespace
    API_KEY = API_KEY.strip().strip('"').strip("'")
else:
    print("WARNING: No WEATHERAPI_KEY found in environment or .env — weather requests may fail or use default values.")

# Construct the WeatherAPI URL (using forecast endpoint, which supports 3 days free)
# Note: For dates far in the future, the free tier may not work, requiring date modification.
weather_url = (
    f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={TRACK_LAT_LON}&days=5&aqi=no&alerts=no"
)

# Fetch and extract weather data
rain_probability = 0
temperature = 20
forecast_data = None
weather_date = resolve_weather_date()

try:
    response = requests.get(weather_url)
    response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
    weather_data = response.json()
    
    # Check if forecast data exists
    if 'forecast' in weather_data and 'forecastday' in weather_data['forecast']:
        available_forecasts = weather_data['forecast']['forecastday']
        available_dates = [day.get('date') for day in available_forecasts if day.get('date')]

        # If chosen date is unavailable, fall back to the first date returned by API.
        if weather_date not in available_dates and available_dates:
            fallback_date = available_dates[0]
            print(
                f"INFO: Weather date {weather_date} unavailable. "
                f"Using nearest available date {fallback_date}."
            )
            weather_date = fallback_date

        # Find the forecast day that matches the race date
        race_day_forecast = next(
            (day for day in available_forecasts if day['date'] == weather_date),
            None
        )

        if race_day_forecast and 'hour' in race_day_forecast:
            # Find the hourly forecast data for 15:00
            race_hour_data = next(
                (hour_data for hour_data in race_day_forecast['hour'] if datetime.strptime(hour_data['time'], '%Y-%m-%d %H:%M').hour == RACE_HOUR),
                None
            )
            
            if race_hour_data:
                # Extract temperature (in Celsius) and chance of rain (as a percentage)
                temperature = race_hour_data.get('temp_c', 20)
                rain_probability = race_hour_data.get('chance_of_rain', 0) / 100 # Convert percent to 0-1
                print(
                    f"WeatherAPI SUCCESS ({weather_date} {RACE_HOUR:02d}:00): "
                    f"Temp={temperature}°C, Rain Prob={rain_probability*100:.0f}%"
                )
            else:
                print(
                    f"WARNING: Could not find {RACE_HOUR:02d}:00 forecast for {weather_date}. "
                    "Using default weather values."
                )
        else:
             print(f"WARNING: Forecast data missing for {weather_date}. Using default weather values.")
    else:
        print("WARNING: WeatherAPI response error. Check API key and plan. Using default weather values.")

except requests.exceptions.RequestException as e:
    print(f"ERROR: Failed to connect to WeatherAPI. Using default weather data. Error: {e}")


# Apply weather adjustment
if rain_probability >= 0.75:
    qualifying_target["QualifyingTime"] = qualifying_target["QualifyingTime (s)"] * 1.05
else:
    qualifying_target["QualifyingTime"] = qualifying_target["QualifyingTime (s)"]

# --- STEP 4: PREPARE FEATURES ---

default_clean_air = training_data["CleanAirRacePace (s)"].median()
default_sector_total = training_data["TotalSectorTime (s)"].median()
default_team_score = training_data["TeamPerformanceScore"].median() if not training_data.empty else 0.5

qualifying_target["CleanAirRacePace (s)"] = qualifying_target["CleanAirRacePace (s)"].fillna(
    qualifying_target["Driver"].map(driver_hist_features["CleanAirRacePace (s)"])
)
qualifying_target["TeamPerformanceScore"] = qualifying_target["TeamPerformanceScore"].fillna(
    qualifying_target["Driver"].map(driver_hist_features["TeamPerformanceScore"])
)

qualifying_target["CleanAirRacePace (s)"] = qualifying_target["CleanAirRacePace (s)"].fillna(
    qualifying_target["Team"].map(team_hist_features["CleanAirRacePace (s)"])
)
qualifying_target["TeamPerformanceScore"] = qualifying_target["TeamPerformanceScore"].fillna(
    qualifying_target["Team"].map(team_hist_features["TeamPerformanceScore"])
)

qualifying_target["CleanAirRacePace (s)"] = qualifying_target["CleanAirRacePace (s)"].fillna(default_clean_air)
qualifying_target["TeamPerformanceScore"] = qualifying_target["TeamPerformanceScore"].fillna(default_team_score)

# merge and create final features
merged_data = qualifying_target.copy()
merged_data["TotalSectorTime (s)"] = merged_data["Driver"].map(ref_sector_map)
merged_data["TotalSectorTime (s)"] = merged_data["TotalSectorTime (s)"].fillna(
    merged_data["Driver"].map(driver_hist_features["TotalSectorTime (s)"])
)
merged_data["TotalSectorTime (s)"] = merged_data["TotalSectorTime (s)"].fillna(
    merged_data["Team"].map(team_hist_features["TotalSectorTime (s)"])
)
merged_data["TotalSectorTime (s)"] = merged_data["TotalSectorTime (s)"].fillna(default_sector_total)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

# --- STEP 5: TRAIN AND PREDICT ---
FEATURE_COLS = [
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "TotalSectorTime (s)",
    "CleanAirRacePace (s)", "QualifyingRank"
]

training_data = training_data.dropna(subset=["TargetLapTime (s)"])

event_keys = (
    training_data[["EventYear", "EventRound"]]
    .drop_duplicates()
    .sort_values(["EventYear", "EventRound"])
    .reset_index(drop=True)
)

if len(event_keys) < 3:
    raise RuntimeError("Need at least 3 races to run realistic event-based validation.")

holdout_events = max(1, int(round(len(event_keys) * 0.2)))
holdout_keys = event_keys.tail(holdout_events)

training_data = training_data.copy()
training_data["EventKey"] = (
    training_data["EventYear"].astype(str) + "-" + training_data["EventRound"].astype(str)
)
holdout_set = set(holdout_keys["EventYear"].astype(str) + "-" + holdout_keys["EventRound"].astype(str))

train_rows = training_data[~training_data["EventKey"].isin(holdout_set)]
test_rows = training_data[training_data["EventKey"].isin(holdout_set)]

if train_rows.empty or test_rows.empty:
    raise RuntimeError("Event split failed. Adjust TRAIN_RACES to include more races.")

X_train_full = train_rows[FEATURE_COLS]
y_train_full = train_rows["TargetLapTime (s)"]
X_test_full = test_rows[FEATURE_COLS]
y_test_full = test_rows["TargetLapTime (s)"]

X = merged_data[FEATURE_COLS]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_full)
X_test_imputed = imputer.transform(X_test_full)
X_imputed = imputer.transform(X)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=38)
model.fit(X_train_imputed, y_train_full)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# --- STEP 6: OUTPUT RESULTS ---
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print(f"\n🏁 Predicted {TARGET_YEAR} {TARGET_EVENT} GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test_imputed)
print(f"Backtest MAE (event split): {mean_absolute_error(y_test_full, y_pred):.2f} seconds")

# --- STEP 7: PLOT ---
plt.figure(figsize=(9, 6))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title(f"Effect of Clean Air Pace on Predicted {TARGET_YEAR} {TARGET_EVENT} GP Results")
plt.tight_layout()
plt.show()