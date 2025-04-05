import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import requests_cache
import openmeteo_requests
from retry_requests import retry
import joblib
import requests
import rasterio
from io import BytesIO
import numpy as np
import ee  # Google Earth Engine
import time
from datetime import datetime
import pickle

# 🔹 Authenticate & Initialize Google Earth Engine
try:
    ee.Authenticate()
    ee.Initialize()
except Exception as e:
    st.error(f"Error initializing Google Earth Engine: {str(e)}")

# Setup Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Load dataset
@st.cache_data
def load_data():
    file_path = "nasa_global_landslide_catalog_point.csv"  # Ensure correct file path
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces
    
    # Extract event year
    if 'date' in df.columns:
        df['event_year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    elif 'event_date' in df.columns:
        df['event_year'] = pd.to_datetime(df['event_date'], errors='coerce').dt.year

    # Drop rows with missing values
    df = df.dropna(subset=['latitude', 'longitude', 'event_year', 'country_name'])

    # Filter only India data
    df = df[df['country_name'].str.lower() == 'india']

    return df

df = load_data()

# Streamlit UI
st.title("🇮🇳 Landslide Risk Assessment Tool (India Only)")
st.write("This app visualizes landslide-prone areas in India using NASA's Global Landslide Catalog dataset.")

# Sidebar filters
st.sidebar.header("Filters")

# Year range filter (2000-2023)
year_range = st.sidebar.slider("Select Year Range", 2000, 2023, (2000, 2023))
df_filtered = df[(df['event_year'] >= year_range[0]) & (df['event_year'] <= year_range[1])]

# State filter (if state column exists)
if 'admin_division_name' in df.columns:  
    df_filtered['admin_division_name'] = df_filtered['admin_division_name'].fillna("Unknown")  # Handle missing states
    states_list = sorted(df_filtered['admin_division_name'].unique())
    selected_state = st.sidebar.selectbox("Select a State", options=["All"] + states_list)

    # Apply state filter
    if selected_state != "All":
        df_filtered = df_filtered[df_filtered['admin_division_name'] == selected_state]

# Base World Map (keeping all locations visible)
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=3, tiles='OpenStreetMap')
marker_cluster = MarkerCluster().add_to(m)

# Adding markers for filtered Indian locations
for _, row in df_filtered.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Date: {row.get('event_date', 'Unknown')}\nState: {row.get('admin_division_name', 'Unknown')}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(marker_cluster)

# Display map
folium_static(m)

# Display filtered dataset
st.write(f"### Filtered Landslide Events ({year_range[0]}-{year_range[1]}, India)")
st.dataframe(df_filtered[['event_date', 'latitude', 'longitude', 'event_year', 'admin_division_name']])

# Function to fetch climate data with improved error handling
def fetch_climate_data(lat, lon, start_date, end_date, max_retries=5, wait_time=5):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "precipitation_hours", "et0_fao_evapotranspiration"],
        "timezone": "Asia/Kolkata",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
            if response.status_code == 429:  
                st.warning(f"⚠️ Rate limit hit! Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if "daily" in data and "time" in data["daily"]:
                return pd.DataFrame({
                    "date": data["daily"]["time"],
                    "temperature_2m_mean": data["daily"]["temperature_2m_mean"],
                    "precipitation_sum": data["daily"]["precipitation_sum"],
                    "rain_sum": data["daily"]["rain_sum"],
                    "precipitation_hours": data["daily"]["precipitation_hours"],
                    "et0_fao_evapotranspiration": data["daily"]["et0_fao_evapotranspiration"],
                })
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching climate data for {lat}, {lon}: {e}")
        
        st.info(f"Retrying ({attempt + 1}/{max_retries}) after {wait_time} seconds...")
        time.sleep(wait_time)

    st.error(f"❌ Failed to fetch climate data for {lat}, {lon} after {max_retries} attempts.")
    return None

# Function to fetch elevation and slope using Google Earth Engine
def fetch_elevation_slope(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem)
        
        elevation = dem.sample(point, 30).first().get("elevation").getInfo()
        slope_value = slope.sample(point, 30).first().get("slope").getInfo()

        return elevation, slope_value
    except Exception as e:
        st.error(f"⚠️ Error fetching elevation & slope for {lat}, {lon}: {e}")
        return None, None

# Interactive location selection
locations = [
    f"{row['latitude']}, {row['longitude']}, {row.get('admin_division_name', 'Unknown')}"
    for _, row in df_filtered.iterrows()
]
clicked_location = st.selectbox("Select a Location", options=["None"] + locations)

if clicked_location and clicked_location != "None":
    lat, lon, state = clicked_location.split(", ")
    latitude = float(lat)
    longitude = float(lon)
    
    # Display details
    st.write(f"**Selected Location Details:**")
    st.write(f"Latitude: {latitude}")
    st.write(f"Longitude: {longitude}")
    st.write(f"State: {state}")
    st.write(f"Selected Year Range: {year_range[0]} - {year_range[1]}")
    
    # Update map to zoom into selected location
    m = folium.Map(location=[latitude, longitude], zoom_start=6, tiles='OpenStreetMap')
    folium.Marker([latitude, longitude], popup=f"Location: {latitude}, {longitude}, {state}",
                  icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    
    # Display updated map
    folium_static(m)
    
    # Fetch Climate Data Button
    if st.button("Fetch Climate Data"):
        climate_data = fetch_climate_data(latitude, longitude, f"{year_range[0]}-01-01", f"{year_range[1]}-12-31")
        st.write(f"**Climate Data for {latitude}, {longitude} (Year Range: {year_range[0]} - {year_range[1]})**")
        st.dataframe(climate_data)

# Add new section for landslide probability prediction
st.markdown("---")
st.title("🔍 Landslide Probability Prediction")

# Function to prepare data for prediction
def prepare_prediction_data(latitude, longitude, elevation, slope, 
                          temperature_mean, precipitation_sum, rain_sum, 
                          precipitation_hours, evapotranspiration, land_cover,
                          year, month, day):
    """
    Prepare data for landslide prediction using the same preprocessing steps
    as during model training.
    
    Returns a DataFrame ready for prediction.
    """
    # Create a DataFrame with the input data
    data = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "elevation": [elevation],
        "slope": [slope],
        "temperature_2m_mean": [temperature_mean],
        "precipitation_sum": [precipitation_sum],
        "rain_sum": [rain_sum],
        "precipitation_hours": [precipitation_hours],
        "et0_fao_evapotranspiration": [evapotranspiration],
        "land_cover": [land_cover],
        "year": [year],
        "month": [month],
        "day": [day]
    })
    
    # Load clustering components
    cluster_preprocessor = joblib.load("cluster_preprocessor.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    
    # Features used for clustering
    cluster_features = ["latitude", "longitude", "elevation", "slope"]
    
    # Apply clustering
    cluster_data = data[cluster_features]
    cluster_data_processed = cluster_preprocessor.transform(cluster_data)
    data["cluster"] = kmeans.predict(cluster_data_processed)
    
    return data

# Load the trained model
@st.cache_resource
def load_model():
    # Load the complete pipeline
    pipeline = joblib.load("landslide_model_pipeline.pkl")
    return pipeline

model_pipeline = load_model()

# Add map for location selection
st.write("Select a location on the map to predict landslide probability:")

# Create a map centered on India
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')

# Add click event to map
m.add_child(folium.LatLngPopup())

# Display the map
folium_static(m, height=400)

# Add input fields for coordinates
st.write("Or enter coordinates manually:")
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.5937)
with col2:
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.9629)

if st.button("Predict Landslide Risk"):
    st.write(f"Selected Location: {latitude}, {longitude}")
    
    with st.spinner("Fetching data and calculating risk..."):
        # Fetch elevation and slope data using Google Earth Engine
        elevation, slope = fetch_elevation_slope(latitude, longitude)
        if elevation is not None and slope is not None:
            st.write(f"Elevation: {elevation:.2f} meters")
            st.write(f"Slope: {slope:.2f} degrees")
        else:
            st.error("Could not fetch terrain data")
            elevation = 0
            slope = 0

        # Fetch climate data
        climate_data = fetch_climate_data(
            latitude, 
            longitude, 
            f"{year_range[0]}-01-01", 
            f"{year_range[1]}-12-31"
        )
        
        if climate_data is not None and not climate_data.empty:
            # Calculate average climate values
            avg_climate = {
                "temperature_2m_mean": climate_data["temperature_2m_mean"].mean(),
                "precipitation_sum": climate_data["precipitation_sum"].sum(),
                "rain_sum": climate_data["rain_sum"].sum(),
                "precipitation_hours": climate_data["precipitation_hours"].sum(),
                "et0_fao_evapotranspiration": climate_data["et0_fao_evapotranspiration"].mean()
            }

            # Prepare data using the helper function
            input_data = prepare_prediction_data(
                latitude=latitude,
                longitude=longitude,
                elevation=elevation,
                slope=slope,
                temperature_mean=avg_climate["temperature_2m_mean"],
                precipitation_sum=avg_climate["precipitation_sum"],
                rain_sum=avg_climate["rain_sum"],
                precipitation_hours=avg_climate["precipitation_hours"],
                evapotranspiration=avg_climate["et0_fao_evapotranspiration"],
                land_cover=1,  # Default value
                year=year_range[1],  # Use the end year from the range
                month=6,       # Default to monsoon season
                day=15         # Mid-month
            )

            # Make prediction using the pipeline
            try:
                probability = model_pipeline.predict_proba(input_data)[0][1] * 100
                risk_category = "Low" if probability < 40 else "Medium" if probability < 70 else "High"
                
                st.write("## 🛑 Landslide Risk Prediction")
                st.metric(label="Predicted Landslide Risk", value=f"{probability:.2f}%", delta=None)
                
                if risk_category == "Low":
                    st.success(f"Risk Category: **{risk_category}**")
                elif risk_category == "Medium":
                    st.warning(f"⚠️ Risk Category: **{risk_category}**")
                else:
                    st.error(f"🚨 Risk Category: **{risk_category}** (Take Precautions!)")
                
                # Display risk factors
                st.write("### Risk Factors Analysis")
                risk_factors = {
                    "Elevation": "High" if elevation > 1000 else "Medium" if elevation > 500 else "Low",
                    "Slope": "High" if slope > 30 else "Medium" if slope > 15 else "Low",
                    "Precipitation": "High" if avg_climate["precipitation_sum"] > 1000 else "Medium" if avg_climate["precipitation_sum"] > 500 else "Low"
                }
                
                for factor, risk in risk_factors.items():
                    st.write(f"- {factor}: {risk} Risk")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check if all required features are present and in the correct order.")
                # Print the actual feature names from the model for debugging
                try:
                    st.write("Model's expected feature names:")
                    st.write(model_pipeline.feature_names_in_)
                    st.write("\nOur input features:")
                    st.write(input_data.tolist())
                except:
                    st.write("Could not get model's feature names")
        else:
            st.error("Could not fetch climate data for the selected location")
