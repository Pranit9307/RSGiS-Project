import pandas as pd
import requests
import time
from datetime import datetime

# -------------------------------
# Function to fetch climate data for a specific date
# -------------------------------
def fetch_climate_data(lat, lon, date, max_retries=3, wait_time=2):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_mean,precipitation_sum,rain_sum,precipitation_hours,et0_fao_evapotranspiration",
        "timezone": "Asia/Kolkata"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            daily = data.get("daily", {})
            if daily:
                # Get the first (and only) day's values
                temperature_2m_mean = daily.get("temperature_2m_mean", [None])[0]
                precipitation_sum = daily.get("precipitation_sum", [None])[0]
                rain_sum = daily.get("rain_sum", [None])[0]
                precipitation_hours = daily.get("precipitation_hours", [None])[0]
                et0 = daily.get("et0_fao_evapotranspiration", [None])[0]
                
                return {
                    "temperature_2m_mean": temperature_2m_mean,
                    "precipitation_sum": precipitation_sum,
                    "rain_sum": rain_sum,
                    "precipitation_hours": precipitation_hours,
                    "et0_fao_evapotranspiration": et0,
                }
        except Exception as e:
            print(f"⚠️ Error fetching climate data for ({lat}, {lon}, {date}): {e}")
            time.sleep(wait_time)
    return None

# -------------------------------------------
# Function to fetch elevation (and slope) data
# -------------------------------------------
def fetch_elevation_slope(lat, lon, max_retries=3, wait_time=2):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            elevation = data.get("elevation", None)
            slope = data.get("slope", None)  # Note: Not all endpoints return slope.
            return elevation, slope
        except Exception as e:
            print(f"⚠️ Error fetching elevation for ({lat}, {lon}): {e}")
            time.sleep(wait_time)
    return None, None

# -------------------------------
# Main script to process the CSV
# -------------------------------

# Read input CSV
input_file = "landslide_risk_training_data2.csv"
df = pd.read_csv(input_file)

# Parse and format the date column (assuming it is in a recognizable format)
# This will convert the date to the format YYYY-MM-DD, which the API requires.
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Drop rows with missing latitude, longitude, or date
df = df.dropna(subset=["latitude", "longitude", "date"])

# Collect results
results = []

# Loop through each row to fetch data for that specific date and location
for index, row in df.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    date = row["date"]

    # Fetch climate data for the specific date
    climate = fetch_climate_data(lat, lon, date)
    # Fetch elevation and slope
    elevation, slope = fetch_elevation_slope(lat, lon)

    if climate:
        results.append({
            "date": date,
            "latitude": lat,
            "longitude": lon,
            "country_name": row.get("country_name", None),
            "admin_division_name": row.get("admin_division_name", None),
            "risk_class": row.get("risk_class", None),
            "elevation": elevation,
            "slope": slope,
            "temperature_2m_mean": climate.get("temperature_2m_mean"),
            "precipitation_sum": climate.get("precipitation_sum"),
            "rain_sum": climate.get("rain_sum"),
            "precipitation_hours": climate.get("precipitation_hours"),
            "et0_fao_evapotranspiration": climate.get("et0_fao_evapotranspiration"),
        })
    else:
        print(f"⚠️ No climate data for ({lat}, {lon}, {date})")

# Save the enriched data to a new CSV
output_file = "climate_terrain_data2.csv"
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"✅ Climate and terrain data saved to '{output_file}' with {len(results)} rows.")
