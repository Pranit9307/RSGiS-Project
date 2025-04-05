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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# Set page config
st.set_page_config(
    page_title="Landslide Risk Assessment Report",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Landslide Risk Assessment Report")
st.markdown("""
Generate detailed risk assessment reports for specific locations.
Includes risk factors analysis, historical context, and mitigation recommendations.
""")

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
    # Input parameters
    st.sidebar.header("Location Details")
    
    latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 0.0)
    longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 0.0)
    elevation = st.sidebar.number_input("Elevation (m)", 0.0, 10000.0, 100.0)
    slope = st.sidebar.number_input("Slope (degrees)", 0.0, 90.0, 15.0)
    
    # Date range for historical analysis
    st.sidebar.header("Analysis Period")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    if st.sidebar.button("Generate Report"):
        # Prepare data for prediction
        input_df = prepare_data(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            slope=slope,
            temperature_mean=25.0,  # Example value
            precipitation_sum=100.0,  # Example value
            rain_sum=80.0,  # Example value
            precipitation_hours=24,  # Example value
            evapotranspiration=5.0,  # Example value
            land_cover=1,
            year=end_date.year,
            month=end_date.month,
            day=end_date.day
        )
        
        # Make prediction
        probability = model.predict_proba(input_df)[0][1] * 100
        risk_category = "Low" if probability < 40 else "Medium" if probability < 70 else "High"
        
        # Create report content
        report_content = []
        
        # Title
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        report_content.append(Paragraph("Landslide Risk Assessment Report", title_style))
        
        # Location Information
        report_content.append(Paragraph("Location Information", styles['Heading2']))
        location_data = [
            ["Latitude", f"{latitude}°"],
            ["Longitude", f"{longitude}°"],
            ["Elevation", f"{elevation}m"],
            ["Slope", f"{slope}°"],
            ["Analysis Date", datetime.now().strftime("%Y-%m-%d")]
        ]
        
        location_table = Table(location_data, colWidths=[200, 200])
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        report_content.append(location_table)
        report_content.append(Spacer(1, 20))
        
        # Risk Assessment
        report_content.append(Paragraph("Risk Assessment", styles['Heading2']))
        risk_data = [
            ["Risk Category", risk_category],
            ["Probability", f"{probability:.2f}%"],
            ["Assessment Date", datetime.now().strftime("%Y-%m-%d")]
        ]
        
        risk_table = Table(risk_data, colWidths=[200, 200])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        report_content.append(risk_table)
        report_content.append(Spacer(1, 20))
        
        # Risk Factors
        report_content.append(Paragraph("Risk Factors Analysis", styles['Heading2']))
        risk_factors = [
            ["Factor", "Value", "Risk Level"],
            ["Elevation", f"{elevation}m", "High" if elevation > 1000 else "Medium" if elevation > 500 else "Low"],
            ["Slope", f"{slope}°", "High" if slope > 30 else "Medium" if slope > 15 else "Low"],
            ["Precipitation", "100mm", "High" if 100 > 1000 else "Medium" if 100 > 500 else "Low"]
        ]
        
        factors_table = Table(risk_factors, colWidths=[150, 100, 100])
        factors_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        report_content.append(factors_table)
        report_content.append(Spacer(1, 20))
        
        # Mitigation Recommendations
        report_content.append(Paragraph("Mitigation Recommendations", styles['Heading2']))
        
        recommendations = {
            "High Risk": [
                "Implement immediate evacuation procedures",
                "Install early warning systems",
                "Strengthen slope stabilization measures",
                "Regular monitoring and assessment"
            ],
            "Medium Risk": [
                "Develop evacuation plans",
                "Install drainage systems",
                "Regular slope maintenance",
                "Community awareness programs"
            ],
            "Low Risk": [
                "Regular inspections",
                "Maintain drainage systems",
                "Monitor weather conditions",
                "Document any changes"
            ]
        }
        
        for rec in recommendations[risk_category]:
            report_content.append(Paragraph(f"• {rec}", styles['Normal']))
        
        # Generate PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        doc.build(report_content)
        
        # Create download button
        st.download_button(
            label="Download PDF Report",
            data=buffer.getvalue(),
            file_name=f"landslide_risk_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
        
        # Display report preview
        st.subheader("Report Preview")
        
        # Location map
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=10
        )
        
        # Add marker for the location
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=10,
            color='red',
            fill=True,
            popup=f"Risk Category: {risk_category}<br>Probability: {probability:.2f}%"
        ).add_to(m)
        
        folium_static(m, width=800, height=400)
        
        # Risk factors visualization
        st.subheader("Risk Factors Analysis")
        
        # Create radar chart for risk factors
        factors = ['Elevation', 'Slope', 'Precipitation']
        values = [
            elevation / 1000,  # Normalize elevation
            slope / 90,        # Normalize slope
            100 / 1000        # Normalize precipitation
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=factors,
            fill='toself',
            name='Risk Factors'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Risk Factors Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True) 