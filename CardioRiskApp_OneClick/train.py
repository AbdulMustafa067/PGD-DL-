import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset (Make sure framingham.csv is in this folder!)
df = pd.read_csv('framingham.csv')
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Setup the exact features matching your main assignment code
num_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
cat_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

num_pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])

# 3. Use your optimal hyperparameters
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', max_depth=5, min_samples_split=2, n_estimators=100, random_state=42))
])

# 4. Fit and save natively into this specific folder path
final_model.fit(X_train, y_train)
joblib.dump(final_model, 'cvd_risk_scoring_model.pkl')
print("🎉 SUCCESS: A 100% compatible model file has been generated!")
