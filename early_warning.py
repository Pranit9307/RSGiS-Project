import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import ee
import numpy as np
from datetime import datetime, timedelta
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="Landslide Early Warning System",
    page_icon="⚠️",
    layout="wide"
)

# Title and description
st.title("⚠️ Landslide Early Warning System")
st.markdown("""
This system provides real-time monitoring and alerts for potential landslide risks.
Stay informed about current conditions and receive early warnings for high-risk areas.
""")

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    st.error(f"Error initializing Earth Engine: {str(e)}")

# Load the model and preprocessing function
@st.cache_resource
def load_model():
    try:
        model = joblib.load("landslide_model_pipeline.pkl")
        with open("prepare_prediction_data.pkl", "rb") as f:
            prepare_data = pickle.load(f)
        return model, prepare_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, prepare_data = load_model()

if model is not None and prepare_data is not None:
    # Sidebar for monitoring settings
    st.sidebar.header("Monitoring Settings")
    
    # Alert threshold
    alert_threshold = st.sidebar.slider(
        "Alert Threshold (%)",
        min_value=0,
        max_value=100,
        value=70,
        help="Set the probability threshold for alerts"
    )
    
    # Monitoring interval
    monitoring_interval = st.sidebar.selectbox(
        "Monitoring Interval",
        ["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"],
        help="How often to check for new alerts"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        # Current weather conditions
        st.subheader("Current Weather Conditions")
        
        # Get current date
        current_date = datetime.now()
        
        # Example weather data (replace with real-time data)
        weather_data = {
            "Temperature": "25°C",
            "Precipitation": "10mm",
            "Humidity": "85%",
            "Wind Speed": "15 km/h"
        }
        
        for metric, value in weather_data.items():
            st.metric(metric, value)
        
        # Weather trend
        st.subheader("Weather Trend")
        # Create example trend data
        dates = pd.date_range(start=current_date - timedelta(days=7), end=current_date, freq='D')
        precipitation_data = np.random.normal(10, 2, len(dates))
        fig = px.line(x=dates, y=precipitation_data,
                     title='Precipitation Trend (Last 7 Days)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk alerts
        st.subheader("Active Risk Alerts")
        
        # Example alerts (replace with real alerts)
        alerts = [
            {"location": "Location A", "risk": 85, "status": "High Risk"},
            {"location": "Location B", "risk": 65, "status": "Moderate Risk"},
            {"location": "Location C", "risk": 45, "status": "Low Risk"}
        ]
        
        for alert in alerts:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{alert['location']}**")
                    st.progress(alert['risk'] / 100)
                with col2:
                    st.write(alert['status'])
    
    # Risk map
    st.subheader("Current Risk Map")
    
    # Create a map centered on the region
    m = folium.Map(
        location=[0, 0],  # Replace with actual center coordinates
        zoom_start=8
    )
    
    # Add risk zones to the map
    # Example risk zones (replace with actual data)
    risk_zones = [
        {"lat": 0, "lon": 0, "risk": 85, "radius": 5000},
        {"lat": 0.1, "lon": 0.1, "risk": 65, "radius": 3000},
        {"lat": 0.2, "lon": 0.2, "risk": 45, "radius": 2000}
    ]
    
    for zone in risk_zones:
        color = 'red' if zone['risk'] >= alert_threshold else 'yellow' if zone['risk'] >= 50 else 'green'
        folium.Circle(
            location=[zone['lat'], zone['lon']],
            radius=zone['radius'],
            color=color,
            fill=True,
            popup=f"Risk: {zone['risk']}%"
        ).add_to(m)
    
    # Display the map
    folium_static(m, width=1200, height=600)
    
    # Alert history
    st.subheader("Alert History")
    
    # Example alert history (replace with actual data)
    alert_history = pd.DataFrame({
        'Date': pd.date_range(start=current_date - timedelta(days=7), end=current_date, freq='D'),
        'Location': ['Location A', 'Location B', 'Location C'] * 3,
        'Risk Level': np.random.choice(['High', 'Moderate', 'Low'], 21),
        'Status': np.random.choice(['Active', 'Resolved', 'Monitoring'], 21)
    })
    
    fig = px.scatter(alert_history, x='Date', y='Location',
                    color='Risk Level',
                    size=[1] * len(alert_history),
                    title='Alert History (Last 7 Days)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Action recommendations
    st.subheader("Recommended Actions")
    
    # Example recommendations based on risk levels
    recommendations = {
        "High Risk": [
            "Evacuate the area immediately",
            "Contact emergency services",
            "Monitor weather conditions closely",
            "Prepare emergency supplies"
        ],
        "Moderate Risk": [
            "Stay alert and monitor conditions",
            "Prepare evacuation plan",
            "Secure loose items",
            "Keep emergency supplies ready"
        ],
        "Low Risk": [
            "Monitor weather updates",
            "Review emergency procedures",
            "Check local alerts",
            "Stay informed"
        ]
    }
    
    for risk_level, actions in recommendations.items():
        with st.expander(f"{risk_level} Risk Actions"):
            for action in actions:
                st.write(f"• {action}")
    
    # Notification settings
    st.sidebar.header("Notification Settings")
    
    st.sidebar.checkbox("Enable Email Notifications")
    st.sidebar.checkbox("Enable SMS Notifications")
    st.sidebar.checkbox("Enable Push Notifications")
    
    # Add notification frequency setting
    st.sidebar.selectbox(
        "Notification Frequency",
        ["Immediate", "Hourly", "Daily", "Weekly"],
        help="How often to receive notifications"
    ) 