# Healthcare Disease Prediction System & Clinical UI App

## ABOUT Author 
[NAME:ABDUL MUSTAFA
 PGD: AI HEALTHCARE
 MODULE-2: APPLIED MACHINE LEARNING FOR CLINICAL INSIGHT
 ENROLLMENT NUMBER: PGD/AIMH/DEC25/309600]

## Description
This project implements an automated, clinical-grade Risk Assessment system that targets preventative healthcare insights. Using a dataset derived from the 10-year **Framingham Heart Study**, the core machine learning engine analyzes primary physiological metrics, patient demographics, and medical backgrounds to flag whether an individual presents a high probability of developing Coronary Heart Disease (CHD) over the next 10 years. 

Furthermore, this system bridges the gap between raw backend models and clinical environments by introducing **`app1.py`**—a fully functional web-responsive application engineered via **Streamlit**. Clinicians can dynamically test patient profiles, use a built-in automated KNN imputer for missing biometric cells, and have the app automatically append patient data and results into a local tracking database file (`patient_records.csv`) upon calculation execution.

## ML Algorithm Used
*   **Logistic Regression Classifier**: Selected specifically because of its strict compliance with medical interpretability standards, allowing clinicians to trace categorical probability mapping accurately without dealing with "black-box" biases.

## Dataset Features
*   `age`: Patient chronological age (Years)
*   `male`: Biological Sex (1 = Male, 0 = Female)
*   `cigsPerDay`: Behavioral smoke indexing tracker (Cigarettes per day)
*   `totChol`: Total Cholesterol marker (mg/dL)
*   `sysBP`: Systolic Blood Pressure vital (mmHg)
*   `diaBP`: Diastolic Blood Pressure vital (mmHg)
*   `BMI`: Body Mass Index tracking metrics (kg/m²)
*   `glucose`: Blood glucose biometric counts (mg/dL)
*   `TenYearCHD`: Binary Clinical Target Vector (1 = High Risk, 0 = Low Risk)

## Visualization Explanation
The submission includes a structured **Confusion Matrix Heatmap** (`visualization_screenshot.png`). This matrix isolates model operational performance markers into 4 distinct quadrants: True Negatives (Healthy correctly diagnosed), False Positives, False Negatives (High risk missed), and True Positives (High risk captured). It visualizes clinical triage capabilities clearly.

## Accuracy
*   **Accuracy Score**: ~66.0% (Calibrated purposefully with `class_weight='balanced'` inside Scikit-Learn to aggressively optimize sensitivity and protect high-stakes patients over raw, arbitrary overall guessing).

## How to Run

### 1. Run the Backend Analytics Script
To train the model pipeline, view performance metrics, and save the visualization:
```bash
pip install -r requirements.txt
python main.py
```

### 2. Run the Interactive Clinical Web Application
To boot up the live doctor dashboard and enable automated CSV patient data logging:
```bash
python -m streamlit run app1.py
```
Open a browser tab to `http://localhost:8501` to test interactive profiles.
