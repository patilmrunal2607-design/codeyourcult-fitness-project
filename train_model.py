import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
df = pd.read_csv("Sleep_Health.csv")

# 2. Define Features and Target
TARGET = "Rest Quality" 
FEATURES = [
    "Gender", "Age", "Occupation", "Sleep Duration", 
    "Physical Activity Level", "Stress Level", "BMI Category"
]

X = df[FEATURES]
y = df[TARGET]

# 3. Create Preprocessing Pipeline (Handles text and numbers automatically)
categorical_features = ["Gender", "Occupation", "BMI Category"]
numerical_features = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Create and Train a Classification Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X, y)

# 5. Save the Model
joblib.dump(model, "sleep_model.pkl")
print("Model trained and saved successfully as sleep_model.pkl!")