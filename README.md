# CodeYourCult Fitness Project

## Problem Statement

**Title:** Student Innovation-Ideas that can boost fitness activities and assist in keeping fit.

**Theme:** Fitness & Sports

## About the Project

This project is developed as a solution for the CodeYourCult Hackathon. 
It aims to encourage fitness activities and assist users in maintaining 
a healthy and active lifestyle.

## Key Features

* Nutrition Calculator – Enter your breakfast, lunch, and dinner dishes (Indian cuisine) along with quantity, and instantly get a detailed nutritional breakdown including calories, carbohydrates, protein, fats, sugar, fibre, sodium, calcium, iron, vitamin C, folate, water intake, and cholesterol.
* Smart Dish Matching – Uses TF-IDF (character n-gram) text vectorization and cosine similarity to intelligently match user-typed dish names to the closest entry in the nutrition database, even with typos or partial names.
* Sleep Quality Prediction – A trained machine learning model predicts a user's sleep quality score based on inputs like gender, age, occupation, sleep duration, physical activity level, stress level, and BMI category.
* Workout Performance Prediction – A set of ML models predict expected heart rate, calories burned, workout intensity, and output based on user age, gender, weight, height, and type of exercise.
* Simple Web Interface – Lightweight Flask backend serving a static frontend (HTML/CSS/JS) for easy interaction with all three modules.
## Technologies Used

* Python (Flask) – backend web framework and REST API
* Pandas – data handling and preprocessing (Excel dataset, ML input formatting)
* Scikit-learn – TF-IDF vectorization, cosine similarity for dish matching, and trained ML models for sleep/workout prediction
* Joblib – loading pre-trained sleep and workout prediction models
* HTML/CSS/JavaScript – frontend served via Flask

## How It Works

1. Nutrition Module: The app loads an Indian food nutrition dataset (Indian_Food_Nutrition.xlsx) at startup and cleans/vectorizes dish names using TF-IDF. When a user submits their meals via the /calculate endpoint, the app finds the closest matching dish for each meal type (Breakfast, Lunch, Dinner) using cosine similarity, then computes total nutrition values scaled by the quantity entered.
2. Sleep Module: The /predict_sleep endpoint accepts user lifestyle data (age, occupation, sleep duration, activity level, stress level, BMI category) and feeds it into a pre-trained regression/classification model to output a predicted sleep quality score.
3. Workout Module: The /predict_workout endpoint accepts physical attributes and exercise type, then runs them through separate trained models to predict heart rate, calories burned, workout intensity, and expected output.
4. Frontend: Flask serves index.html and static assets directly, providing a single-page interface to interact with all three features.

## How to Run

1. Clone this repository:
   git clone <https://github.com/patilmrunal2607-design/codeyourcult-fitness-project/tree/main>
   
2. Install the required dependencies:
   pip install flask pandas scikit-learn joblib openpyxl
   
3. Make sure the following files are present in the project root:
   Indian_Food_Nutrition.xlsx (nutrition dataset)
   sleep_model.pkl (trained sleep prediction model)
   workout_model.pkl (trained workout prediction model)
   index.html (frontend page)
   
4. Run the Flask app:
   python app.py
   
5. Open your browser and go to:
   http://127.0.0.1:5000/   

## Future Scope

* Expand the nutrition database to include more regional and international cuisines.
* Add user accounts to track nutrition, sleep, and workout history over time.
* Provide personalized diet and workout recommendations based on predicted sleep and fitness metrics.
* Deploy the application on a cloud platform for public access.

## Team Members

- Rushikesh Sargar
- Premchand Khade
- Mrunal Patil
- Monika Khambale
