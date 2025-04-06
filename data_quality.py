import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import pickle
from scipy import stats
from model_pipeline import prepare_prediction_data

# Set page config
st.set_page_config(
    page_title="Data Quality Monitoring",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Data Quality Monitoring")
st.markdown("""
Monitor and validate the quality of input data and model predictions.
Identify potential issues, outliers, and data inconsistencies.
""")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("landslide_model_pipeline.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is not None:
    # Load historical data
    @st.cache_data
    def load_historical_data():
        try:
            df = pd.read_csv("output\climate_terrain_data2.csv")
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            return df
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            return None

    df = load_historical_data()

    if df is not None:
        # Sidebar filters
        st.sidebar.header("Data Quality Filters")
        
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
        
        # Data Quality Metrics
        st.header("Data Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Missing values percentage
            missing_pct = filtered_df.isnull().mean() * 100
            st.metric("Missing Values", f"{missing_pct.mean():.1f}%")
        
        with col2:
            # Outliers percentage
            numerical_features = filtered_df.select_dtypes(include=[np.number]).columns
            z_scores = stats.zscore(filtered_df[numerical_features])
            outlier_percentage = (abs(z_scores) > 3).mean().mean() * 100
            st.metric("Outliers", f"{outlier_percentage:.1f}%")
        
        with col3:
            # Data completeness
            completeness = (1 - missing_pct.mean()) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col4:
            # Data consistency
            consistency = (1 - outlier_percentage/100) * 100
            st.metric("Data Consistency", f"{consistency:.1f}%")
        
        # Feature Distribution Analysis
        st.header("Feature Distribution Analysis")
        
        # Select feature to analyze
        feature = st.selectbox(
            "Select Feature",
            options=filtered_df.select_dtypes(include=[np.number]).columns
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig = px.histogram(filtered_df, x=feature,
                             title=f"{feature} Distribution",
                             nbins=50)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(filtered_df, y=feature,
                        title=f"{feature} Box Plot")
            st.plotly_chart(fig, use_container_width=True)
        
        # Outlier Detection
        st.header("Outlier Detection")
        
        # Calculate z-scores for numerical features
        numerical_features = filtered_df.select_dtypes(include=[np.number]).columns
        z_scores = stats.zscore(filtered_df[numerical_features])
        outliers = abs(z_scores) > 3
        
        # Create heatmap of outliers
        fig = px.imshow(outliers,
                       title="Outlier Heatmap",
                       labels=dict(x="Features", y="Samples", color="Outlier"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Correlation Analysis
        st.header("Feature Correlation Analysis")
        
        # Calculate correlation matrix
        correlation_matrix = filtered_df[numerical_features].corr()
        
        # Create correlation heatmap
        fig = px.imshow(correlation_matrix,
                       title="Feature Correlation Heatmap",
                       labels=dict(x="Features", y="Features", color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Quality Trends
        st.header("Data Quality Trends")
        
        # Calculate daily quality metrics
        daily_metrics = filtered_df.groupby(filtered_df['date'].dt.date).agg({
            'elevation': lambda x: x.isnull().mean(),
            'slope': lambda x: x.isnull().mean(),
            'temperature_2m_mean': lambda x: x.isnull().mean(),
            'precipitation_sum': lambda x: x.isnull().mean()
        }).mean(axis=1)
        
        # Create trend plot
        fig = px.line(x=daily_metrics.index, y=daily_metrics.values,
                     title="Daily Missing Data Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Validation Rules
        st.header("Data Validation Rules")
        
        # Define validation rules
        validation_rules = {
            "Elevation": {
                "min": 0,
                "max": 10000,
                "unit": "meters"
            },
            "Slope": {
                "min": 0,
                "max": 90,
                "unit": "degrees"
            },
            "Temperature": {
                "min": -50,
                "max": 50,
                "unit": "°C"
            },
            "Precipitation": {
                "min": 0,
                "max": 1000,
                "unit": "mm"
            }
        }
        
        # Check validation rules
        validation_results = {}
        for feature, rules in validation_rules.items():
            if feature in filtered_df.columns:
                violations = (
                    (filtered_df[feature] < rules['min']) |
                    (filtered_df[feature] > rules['max'])
                ).mean() * 100
                validation_results[feature] = violations
        
        # Display validation results
        for feature, violations in validation_results.items():
            st.metric(
                f"{feature} Range Violations",
                f"{violations:.1f}%",
                delta=None,
                help=f"Percentage of values outside valid range ({validation_rules[feature]['min']} to {validation_rules[feature]['max']} {validation_rules[feature]['unit']})"
            )
        
        # Data Quality Report
        st.header("Data Quality Report")
        
        # Generate report content
        report_content = []
        
        # Overall statistics
        report_content.append("### Overall Statistics")
        report_content.append(f"- Total samples: {len(filtered_df)}")
        report_content.append(f"- Date range: {date_range[0]} to {date_range[1]}")
        report_content.append(f"- Missing values: {missing_pct.mean():.1f}%")
        report_content.append(f"- Outliers: {outlier_percentage:.1f}%")
        
        # Feature-specific statistics
        report_content.append("\n### Feature Statistics")
        for feature in numerical_features:
            stats = filtered_df[feature].describe()
            report_content.append(f"\n#### {feature}")
            report_content.append(f"- Mean: {stats['mean']:.2f}")
            report_content.append(f"- Std: {stats['std']:.2f}")
            report_content.append(f"- Min: {stats['min']:.2f}")
            report_content.append(f"- Max: {stats['max']:.2f}")
        
        # Display report
        for content in report_content:
            st.markdown(content) 