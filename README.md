# RSGiS-Project


# Landslide Risk Assessment System - Project Documentation

## Overview
This project is a comprehensive system for assessing landslide risks in India using machine learning and geospatial analysis. It combines historical landslide data, climate information, and terrain characteristics to predict landslide probabilities and visualize risk areas.

## System Architecture

### 1. Data Collection and Processing (`data3.py`)
- **Purpose**: Collects and processes climate and terrain data for training the model
- **Key Features**:
  - Fetches historical climate data from Open-Meteo API
  - Retrieves elevation and slope data
  - Processes and combines multiple data sources
  - Outputs enriched dataset for model training

### 2. Model Training (`model2.py`)
- **Purpose**: Trains the machine learning model for landslide risk prediction
- **Key Features**:
  - Uses RandomForestClassifier for prediction
  - Implements hyperparameter tuning using RandomizedSearchCV
  - Handles categorical variables through label encoding
  - Saves trained model and encoders for deployment

### 3. Web Application (`test2.py`)
- **Purpose**: Interactive web interface for landslide risk assessment
- **Key Features**:
  - Interactive map for location selection
  - Real-time risk prediction
  - Climate data visualization
  - Risk heatmap generation

## Technical Components

### Data Sources
1. **NASA Global Landslide Catalog**
   - Historical landslide events
   - Location data (latitude, longitude)
   - Event dates and severity

2. **Climate Data (Open-Meteo API)**
   - Temperature
   - Precipitation
   - Evapotranspiration
   - Historical weather patterns

3. **Terrain Data**
   - Elevation
   - Slope
   - Land cover information

### Machine Learning Model
- **Algorithm**: RandomForestClassifier
- **Features Used**:
  - Geographic coordinates (latitude, longitude)
  - Elevation and slope
  - Climate variables
  - Administrative division (state)
- **Output**: Risk classification (Low, Medium, High)

### Web Interface Components
1. **Interactive Map**
   - Location selection
   - Risk visualization
   - Heatmap generation

2. **Data Input**
   - Manual coordinate entry
   - Map-based selection
   - Automatic state detection

3. **Results Display**
   - Risk probability
   - Risk factors analysis
   - Climate data visualization

## Implementation Details

### Data Processing Pipeline
1. Data collection from multiple sources
2. Feature engineering and preprocessing
3. Model training and validation
4. Model deployment and prediction

### Risk Assessment Process
1. Location selection (map or manual input)
2. Climate data retrieval
3. Terrain data analysis
4. Risk prediction
5. Results visualization

## Usage Guide

### Setting Up the Environment
1. Install required packages:
   ```bash
   pip install streamlit folium pandas numpy scikit-learn joblib requests
   pip install streamlit-folium openmeteo-requests retry-requests
   ```

2. Set up Google Earth Engine authentication
3. Configure API keys (if required)

### Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run test2.py
   ```

2. Access the web interface through the provided URL

### Using the Application
1. Select a location using the map or manual input
2. View the detected state and coordinates
3. Click "Predict Landslide Risk" to get results
4. Analyze the risk factors and probabilities

## Project Structure
```
project_root/
├── data3.py              # Data collection and processing
├── model2.py             # Model training
├── test2.py              # Web application
├── output/               # Generated files
│   ├── risk_class_rf_model_tuned.pkl
│   ├── risk_class_label_encoder.pkl
│   └── admin_division_label_encoder.pkl
└── data/                 # Input data files
    └── nasa_global_landslide_catalog_point.csv
```

## Dependencies
- Python 3.7+
- Streamlit
- Folium
- Pandas
- NumPy
- scikit-learn
- Joblib
- Requests
- Open-Meteo API
- Google Earth Engine

## Future Enhancements
1. Real-time weather data integration
2. More detailed terrain analysis
3. Historical risk trend visualization
4. Mobile application version
5. Additional risk factors consideration

## Limitations
1. Dependence on external APIs for data
2. Limited to Indian geographical region
3. Historical data constraints
4. Model accuracy limitations

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
[Specify your license here]

## Contact
[Your contact information] 
