import numpy as np
import pandas as pd
import joblib

# For clustering and preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = "resampled_landslide_data4.csv"
df = pd.read_csv(file_path)

# -----------------------------
# Data Preprocessing
# -----------------------------

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Extract year, month, and day as separate columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop the original 'date' column
df = df.drop(columns=['date'])

# Drop 'clay_content' since it contains all missing values
df = df.drop(columns=['clay_content'])

# Define numerical and categorical features
numerical_features = ["latitude", "longitude", "elevation", "slope", 
                      "temperature_2m_mean", "precipitation_sum", "rain_sum", 
                      "precipitation_hours", "et0_fao_evapotranspiration",
                      "year", "month", "day"]
categorical_features = ["land_cover"]

# Create preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# -----------------------------
# Clustering
# -----------------------------

# Features used for clustering
cluster_features = ["latitude", "longitude", "elevation", "slope"]
X_cluster = df[cluster_features].copy()

# Create separate pipeline for clustering
cluster_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_cluster_processed = cluster_preprocessor.fit_transform(X_cluster)

# Perform KMeans clustering with more clusters for better spatial differentiation
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_cluster_processed)

print("Cluster counts:")
print(df["cluster"].value_counts())

# Save cluster model and preprocessor for later use
joblib.dump(cluster_preprocessor, "cluster_preprocessor.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")

# -----------------------------
# Prepare Data for Classification
# -----------------------------

# Define all features including the cluster
all_features = numerical_features + categorical_features + ["cluster"]

X = df[all_features]
y = df["landslide_occurred"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Define the complete model pipeline
# -----------------------------

# Create the classification pipeline
classification_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', None)  # Placeholder to be filled later
])

# -----------------------------
# Hyperparameter Tuning for RandomForest
# -----------------------------

rf = RandomForestClassifier(random_state=42)
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 15, 30],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

# Set up the pipeline with RandomForest for grid search
classification_pipeline.set_params(classifier=rf)

grid_search = GridSearchCV(classification_pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("Best RandomForest Parameters:", grid_search.best_params_)

# -----------------------------
# Ensemble Learning via Stacking
# -----------------------------

# Base learners: Best RandomForest and SVM (with probability estimates)
estimators = [
    ('rf', RandomForestClassifier(**{k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}, random_state=42)),
    ('svm', SVC(probability=True, random_state=42, kernel='rbf', C=1.0))
]

# Meta-learner: Logistic Regression with balanced class weights
stack_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    n_jobs=-1
)

# Set up the pipeline with the stacking classifier
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stack_clf)
])

# Train final model
final_pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluation on Test Set
# -----------------------------

y_pred = final_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble Model Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Feature Importance Analysis
# -----------------------------

# Extract feature importance from the Random Forest base learner
rf_classifier = final_pipeline.named_steps['classifier'].estimators_[0][1]
feature_importance = pd.DataFrame(
    rf_classifier.feature_importances_,
    index=preprocessor.get_feature_names_out(),
    columns=['importance']
).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# -----------------------------
# Save the complete model with preprocessing for later use
# -----------------------------

# Save the entire pipeline
joblib.dump(final_pipeline, "landslide_model_pipeline.pkl")

# Save a utility function for prediction to ensure consistent preprocessing
import pickle

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
    
    # Apply clustering
    cluster_data = data[cluster_features]
    cluster_data_processed = cluster_preprocessor.transform(cluster_data)
    data["cluster"] = kmeans.predict(cluster_data_processed)
    
    return data

# Save the prepare_prediction_data function
with open("prepare_prediction_data.pkl", "wb") as f:
    pickle.dump(prepare_prediction_data, f)

print("Complete model pipeline saved as landslide_model_pipeline.pkl")
print("Helper function saved as prepare_prediction_data.pkl")
