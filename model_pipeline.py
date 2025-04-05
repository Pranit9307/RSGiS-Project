import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import joblib
import pickle

def prepare_prediction_data(latitude, longitude, elevation, slope, 
                          temperature_mean, precipitation_sum, rain_sum,
                          precipitation_hours, evapotranspiration, land_cover,
                          year, month, day):
    """
    Prepare input data for prediction with the correct feature order.
    """
    # Create input data dictionary
    input_data = {
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation,
        'slope': slope,
        'temperature_2m_mean': temperature_mean,
        'precipitation_sum': precipitation_sum,
        'rain_sum': rain_sum,
        'precipitation_hours': precipitation_hours,
        'et0_fao_evapotranspiration': evapotranspiration,
        'year': year,
        'month': month,
        'day': day,
        'land_cover': land_cover,
        'cluster': 0  # Default cluster value
    }
    
    # Create DataFrame with specific feature order
    feature_order = [
        'latitude', 'longitude', 'elevation', 'slope',
        'temperature_2m_mean', 'precipitation_sum', 'rain_sum',
        'precipitation_hours', 'et0_fao_evapotranspiration',
        'year', 'month', 'day', 'land_cover', 'cluster'
    ]
    
    df = pd.DataFrame([input_data])
    df = df[feature_order]
    
    return df

def create_model_pipeline():
    """
    Create and return the complete model pipeline.
    """
    # Define numerical and categorical features
    numerical_features = [
        'latitude', 'longitude', 'elevation', 'slope',
        'temperature_2m_mean', 'precipitation_sum', 'rain_sum',
        'precipitation_hours', 'et0_fao_evapotranspiration',
        'year', 'month', 'day'
    ]
    
    categorical_features = ['land_cover', 'cluster']
    
    # Create preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the complete pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    return model_pipeline

def save_model_pipeline(model_pipeline, prepare_data_func):
    """
    Save the model pipeline and preprocessing function.
    """
    # Save the model pipeline
    joblib.dump(model_pipeline, "landslide_model_pipeline.pkl")
    
    # Save the preprocessing function
    with open("prepare_prediction_data.pkl", "wb") as f:
        pickle.dump(prepare_data_func, f)

if __name__ == "__main__":
    # Create and save the model pipeline
    model_pipeline = create_model_pipeline()
    save_model_pipeline(model_pipeline, prepare_prediction_data) 