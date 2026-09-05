import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the workout dataset
df = pd.read_excel('exercises4 - Copy.csv.xlsx')

# Clean text columns to ensure they match user inputs perfectly
df['type of exercise'] = df['type of exercise'].str.upper().str.strip()
df['gender'] = df['gender'].str.upper().str.strip()

# 2. Define Features
X = df[['age', 'gender', 'weight (KG)', 'height (CM)', 'type of exercise']]

# 3. Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['age', 'weight (KG)', 'height (CM)']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'type of exercise'])
    ])

# 4. Create Models for the 4 Targets
hr_model = Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
cal_model = Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
int_model = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
out_model = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

# 5. Train all models
print("Training models... This might take a few seconds.")
hr_model.fit(X, df['heart rate (BPM)'])
cal_model.fit(X, df['calories burned'])
int_model.fit(X, df['intensity level'])
out_model.fit(X, df['OUTPUT'])

# 6. Save them as a single dictionary package
workout_models = {
    'hr': hr_model,
    'cal': cal_model,
    'intensity': int_model,
    'output': out_model
}

joblib.dump(workout_models, "workout_model.pkl")
print("Workout models trained and saved successfully as workout_model.pkl!")