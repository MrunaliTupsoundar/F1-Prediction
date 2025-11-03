import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
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

# --- STEP 1: LOAD 2024 TRAINING DATA (AZERBAIJAN GP) ---
session_2024 = fastf1.get_session(2024, "Azerbaijan", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# --- STEP 2: DEFINE 2025 PREDICTION FEATURES ---

# ⚠️ PLACEHOLDER: Clean air race pace (MUST BE UPDATED FOR BAKU 2024 RACE!)
clean_air_race_pace = {
    "VER": 97.500, "HAM": 98.200, "LEC": 97.650, "NOR": 97.800, "ALO": 99.000,
    "PIA": 97.750, "RUS": 98.300, "SAI": 98.700, "STR": 99.400, "HUL": 99.800,
    "OCO": 100.100, "TSU": 99.600, "GAS": 100.300
}

# ⚠️ PLACEHOLDER: 2025 Baku Qualifying Data (MUST BE UPDATED WITH ACTUAL 2025 RESULTS)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
               "TSU", "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [101.500, 101.650, 101.800, 101.900, 102.100, 102.300,
                           102.350, 102.450, 102.600, 102.800, 103.000, 103.100, 103.200, 103.300]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# --- STEP 3: WEATHER DATA (USING WEATHERAPI.COM) ---

# Read Weather API key from environment (or .env loaded above)
API_KEY = os.environ.get("WEATHERAPI_KEY") or os.environ.get("API_KEY") or ""
if API_KEY:
    # sanitize common cases where the value might include surrounding quotes or whitespace
    API_KEY = API_KEY.strip().strip('"').strip("'")
else:
    print("WARNING: No WEATHERAPI_KEY found in environment or .env — weather requests may fail or use default values.")

# Baku coordinates and race date/time (RACE_DATE / RACE_HOUR can be overridden via .env)
BAKU_LAT_LON = "40.3725,49.8533"
RACE_DATE = os.environ.get("RACE_DATE", "2025-11-03")
try:
    RACE_HOUR = int(os.environ.get("RACE_HOUR", "15"))
except ValueError:
    RACE_HOUR = 15 # Race starts at 15:00 local time by default

# Construct the WeatherAPI URL (using forecast endpoint, which supports 3 days free)
# Note: For dates far in the future, the free tier may not work, requiring date modification.
weather_url = (
    f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={BAKU_LAT_LON}&days=5&aqi=no&alerts=no"
)

# Fetch and extract weather data
rain_probability = 0
temperature = 20
forecast_data = None

try:
    response = requests.get(weather_url)
    response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
    weather_data = response.json()
    
    # Check if forecast data exists
    if 'forecast' in weather_data and 'forecastday' in weather_data['forecast']:
        # Find the forecast day that matches the race date
        race_day_forecast = next(
            (day for day in weather_data['forecast']['forecastday'] if day['date'] == RACE_DATE),
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
                print(f"WeatherAPI SUCCESS: Temp={temperature}°C, Rain Prob={rain_probability*100:.0f}%")
            else:
                print("WARNING: Could not find 15:00 forecast for race day. Using default weather values.")
        else:
             print("WARNING: Forecast data missing for race date. Using default weather values.")
    else:
        print("WARNING: WeatherAPI response error. Check API key and plan. Using default weather values.")

except requests.exceptions.RequestException as e:
    print(f"ERROR: Failed to connect to WeatherAPI. Using default weather data. Error: {e}")


# Apply weather adjustment
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * 1.05
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# --- STEP 4: CONSTRUCTOR DATA AND MERGE ---
# (Using same team data as before)
team_points = {
    "McLaren": 203, "Mercedes": 118, "Red Bull": 92, "Williams": 25, "Ferrari": 84,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# merge and create final features
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["LastYearWinner"] = (merged_data["Driver"] == "VER").astype(int)
merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

# --- STEP 5: TRAIN AND PREDICT ---
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore",
    "CleanAirRacePace (s)"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=38)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=38)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# --- STEP 6: OUTPUT RESULTS ---
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Azerbaijan GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE ): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# --- STEP 7: PLOT ---
plt.figure(figsize=(9, 6))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title("Effect of Clean Air Race Pace on Predicted 2025 Azerbaijan GP Results (WeatherAPI)")
plt.tight_layout()
plt.show()