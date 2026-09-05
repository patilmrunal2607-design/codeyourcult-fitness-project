from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import re

app = Flask(__name__)

# --------------------------------------------------
# 1. NUTRITION SETUP
# --------------------------------------------------
DATASET = "Indian_Food_Nutrition.xlsx"
try:
    df = pd.read_excel(DATASET)
    df = df.dropna(subset=["Dish Name"])
    df["Dish Name"] = df["Dish Name"].astype(str)
    
    def clean_text(text):
        text = str(text).lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text)
        
    df["clean_name"] = df["Dish Name"].apply(clean_text)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    dish_vectors = vectorizer.fit_transform(df["clean_name"])
except FileNotFoundError:
    print(f"Warning: {DATASET} not found.")
    df = pd.DataFrame()

def find_dish(user_input, meal_type):
    user_input = clean_text(user_input)
    if not user_input or df.empty: return None
    meal_df = df[df["Meal Type"].astype(str).str.lower() == meal_type.lower()]
    if len(meal_df) == 0: return None
    
    exact = meal_df[meal_df["clean_name"] == user_input]
    if len(exact) > 0: return exact.iloc[0]
    
    meal_indices = meal_df.index.tolist()
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, dish_vectors[meal_indices])[0]
    best_position = similarities.argmax()
    
    if similarities[best_position] < 0.25: return None
    return df.loc[meal_indices[best_position]]

# --------------------------------------------------
# 2. SLEEP MODEL SETUP
# --------------------------------------------------
try:
    sleep_model = joblib.load("sleep_model.pkl")
except FileNotFoundError:
    sleep_model = None

# --------------------------------------------------
# 3. WORKOUT MODEL SETUP
# --------------------------------------------------
try:
    workout_models = joblib.load("workout_model.pkl")
except FileNotFoundError:
    workout_models = None


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.json
    meals = [
        ("Breakfast", data.get("breakfast", ""), float(data.get("breakfast_qty", 1))),
        ("Lunch", data.get("lunch", ""), float(data.get("lunch_qty", 1))),
        ("Dinner", data.get("dinner", ""), float(data.get("dinner_qty", 1)))
    ]
    selected_dishes, errors = [], []

    for meal_type, dish_name, qty in meals:
        if not dish_name.strip():
            errors.append(f"Please enter a dish for {meal_type}.")
            continue
        dish = find_dish(dish_name, meal_type)
        if dish is None:
            errors.append(f"Could not find '{dish_name}' under {meal_type}.")
            continue
        selected_dishes.append({"meal": meal_type, "dish_data": dish, "qty": qty})

    if errors: return jsonify({"success": False, "errors": errors})

    cols = ["Calories (kcal)", "Carbohydrates (g)", "Protein (g)", "Fats (g)", "Free Sugar (g)", "Fibre (g)", "Sodium (mg)", "Calcium (mg)", "Iron (mg)", "Vitamin C (mg)", "Folate (µg)", "Water Intake per Dish (ml)", "Cholesterol (mg per Dish)"]
    totals = {}
    for c in cols:
        vals = [float(pd.to_numeric(item["dish_data"].get(c, 0), errors="coerce")) * item["qty"] for item in selected_dishes if pd.notna(pd.to_numeric(item["dish_data"].get(c, 0), errors="coerce"))]
        totals[c] = round(sum(vals), 2)

    return jsonify({"success": True, "dishes": [{"meal": i["meal"], "dish": i["dish_data"]["Dish Name"], "qty": i["qty"]} for i in selected_dishes], "nutrition": totals})

@app.route("/predict_sleep", methods=["POST"])
def predict_sleep():
    if not sleep_model: return jsonify({"success": False, "error": "Sleep model not trained."})
    data = request.json
    input_data = pd.DataFrame([{"Gender": data.get("gender"), "Age": float(data.get("age")), "Occupation": data.get("occupation"), "Sleep Duration": float(data.get("duration")), "Physical Activity Level": float(data.get("activity")), "Stress Level": float(data.get("stress")), "BMI Category": data.get("bmi")}])
    prediction = sleep_model.predict(input_data)[0]
    return jsonify({"success": True, "quality_score": prediction})

@app.route("/predict_workout", methods=["POST"])
def predict_workout():
    if not workout_models: return jsonify({"success": False, "error": "Workout model not trained."})
    data = request.json
    
    input_data = pd.DataFrame([{
        "age": float(data.get("age")),
        "gender": data.get("gender").upper(),
        "weight (KG)": float(data.get("weight")),
        "height (CM)": float(data.get("height")),
        "type of exercise": data.get("exercise").upper()
    }])

    return jsonify({
        "success": True,
        "heart_rate": round(workout_models['hr'].predict(input_data)[0]),
        "calories": round(workout_models['cal'].predict(input_data)[0]),
        "intensity": workout_models['intensity'].predict(input_data)[0],
        "output": workout_models['output'].predict(input_data)[0]
    })

@app.route("/")
def home(): return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def serve_file(filename): return send_from_directory(".", filename)

if __name__ == "__main__": app.run(debug=True)