import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import numpy as np
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Historical Landslide Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Historical Landslide Analysis")
st.markdown("""
This dashboard provides insights into historical landslide occurrences and patterns.
Analyze trends, identify high-risk areas, and understand the factors contributing to landslides.
""")

# Load historical data
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("climate data manual.csv")
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

df = load_historical_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
    filtered_df = df[mask]
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series of landslide occurrences
        st.subheader("Landslide Occurrences Over Time")
        daily_occurrences = filtered_df.groupby('date')['landslide_occurred'].sum().reset_index()
        fig = px.line(daily_occurrences, x='date', y='landslide_occurred',
                     title='Daily Landslide Occurrences')
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly patterns
        st.subheader("Monthly Patterns")
        filtered_df['month'] = filtered_df['date'].dt.month
        monthly_occurrences = filtered_df.groupby('month')['landslide_occurred'].mean().reset_index()
        fig = px.bar(monthly_occurrences, x='month', y='landslide_occurred',
                    title='Average Landslide Occurrences by Month')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk factors correlation
        st.subheader("Risk Factors Correlation")
        risk_factors = ['elevation', 'slope', 'precipitation_sum', 'temperature_2m_mean']
        correlation_matrix = filtered_df[risk_factors + ['landslide_occurred']].corr()
        fig = px.imshow(correlation_matrix,
                       title='Correlation between Risk Factors and Landslide Occurrence')
        st.plotly_chart(fig, use_container_width=True)
        
        # Elevation vs Slope scatter plot
        st.subheader("Elevation vs Slope Analysis")
        fig = px.scatter(filtered_df, x='elevation', y='slope',
                        color='landslide_occurred',
                        title='Elevation vs Slope (Color: Landslide Occurrence)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Full-width map
    st.subheader("Historical Landslide Locations")
    
    # Create a map centered on the data
    m = folium.Map(
        location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()],
        zoom_start=8
    )
    
    # Add landslide points to the map
    for idx, row in filtered_df[filtered_df['landslide_occurred'] == 1].iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red',
            fill=True,
            popup=f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                  f"Elevation: {row['elevation']}m<br>"
                  f"Slope: {row['slope']}°"
        ).add_to(m)
    
    # Display the map
    folium_static(m, width=1200, height=600)
    
    # Additional insights
    st.subheader("Key Insights")
    
    # Calculate statistics
    total_landslides = filtered_df['landslide_occurred'].sum()
    avg_elevation = filtered_df[filtered_df['landslide_occurred'] == 1]['elevation'].mean()
    avg_slope = filtered_df[filtered_df['landslide_occurred'] == 1]['slope'].mean()
    avg_precipitation = filtered_df[filtered_df['landslide_occurred'] == 1]['precipitation_sum'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Landslides", f"{total_landslides}")
    with col2:
        st.metric("Average Elevation", f"{avg_elevation:.1f}m")
    with col3:
        st.metric("Average Slope", f"{avg_slope:.1f}°")
    with col4:
        st.metric("Avg Precipitation", f"{avg_precipitation:.1f}mm")
    
    # Risk factor analysis
    st.subheader("Risk Factor Analysis")
    
    # Create risk factor distributions
    risk_factors = ['elevation', 'slope', 'precipitation_sum', 'temperature_2m_mean']
    for factor in risk_factors:
        fig = px.box(filtered_df, y=factor, color='landslide_occurred',
                    title=f'{factor} Distribution by Landslide Occurrence')
        st.plotly_chart(fig, use_container_width=True) 