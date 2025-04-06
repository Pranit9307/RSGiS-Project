import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from scipy.stats import randint

# -------------------------------
# Load the dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\prani\OneDrive\Desktop\Rsgis new\output\preprocessed_climate_terrain_data.csv")

# -------------------------------
# Clean 'elevation' column if needed
# -------------------------------
def clean_elevation(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            return float(val.strip("[]"))
        except:
            return np.nan
    return val

df['elevation'] = df['elevation'].apply(clean_elevation).astype(float)

# -------------------------------
# Drop 'slope' column if it's empty
# -------------------------------
if 'slope' in df.columns and df['slope'].isnull().all():
    df.drop(columns=['slope'], inplace=True)

# -------------------------------
# Drop rows with missing values
# -------------------------------
df.dropna(inplace=True)

# -------------------------------
# Encode categorical variable 'admin_division_name'
# -------------------------------
le_admin = LabelEncoder()
df['admin_division_name'] = le_admin.fit_transform(df['admin_division_name'].astype(str))

# -------------------------------
# Encode target variable 'risk_class'
# -------------------------------
target_encoder = LabelEncoder()
df['risk_class_encoded'] = target_encoder.fit_transform(df['risk_class'])

# -------------------------------
# Define features and target (excluding 'country_name')
# -------------------------------
feature_columns = [
    'latitude', 'longitude', 'admin_division_name', 'elevation',
    'temperature_2m_mean', 'precipitation_sum', 'rain_sum',
    'precipitation_hours', 'et0_fao_evapotranspiration'
]
X = df[feature_columns]
y = df['risk_class_encoded']

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Build a Pipeline with Scaling and RandomForest
# -------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# -------------------------------
# Define Hyperparameter Search Space
# -------------------------------
param_distributions = {
    'clf__n_estimators': randint(100, 1000),
    'clf__max_depth': [None] + list(range(5, 30, 5)),
    'clf__min_samples_split': randint(2, 10),
    'clf__min_samples_leaf': randint(1, 5),
    'clf__max_features': ['auto', 'sqrt', 'log2']
}

# -------------------------------
# Hyperparameter Tuning using RandomizedSearchCV
# -------------------------------
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best Model
best_model = random_search.best_estimator_

# -------------------------------
# Evaluate the Model
# -------------------------------
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=target_encoder.classes_))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Save the best model and encoders
# -------------------------------
joblib.dump(best_model, r"C:\Users\prani\OneDrive\Desktop\Rsgis new\output\risk_class_rf_model_tuned.pkl")
joblib.dump(target_encoder, r"C:\Users\prani\OneDrive\Desktop\Rsgis new\output\risk_class_label_encoder.pkl")
joblib.dump(le_admin, r"C:\Users\prani\OneDrive\Desktop\Rsgis new\output\admin_division_label_encoder.pkl")
print("💾 Tuned model and encoders saved.")
