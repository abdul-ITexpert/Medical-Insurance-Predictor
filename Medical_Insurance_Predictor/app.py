from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/catboost_model.pkl")




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    try:

        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = request.form["region"]

        northwest = 0
        southeast = 0
        southwest = 0

        if region == "northwest":
            northwest = 1

        elif region == "southeast":
            southeast = 1

        elif region == "southwest":
            southwest = 1

        data = pd.DataFrame(
            [[
                age,
                sex,
                bmi,
                children,
                smoker,
                northwest,
                southeast,
                southwest
            ]],
            columns=[
                "age",
                "sex",
                "bmi",
                "children",
                "smoker",
                "region_northwest",
                "region_southeast",
                "region_southwest"
            ]
        )

        prediction = round(
            float(np.expm1(model.predict(data))[0]),
            2
        )

        return render_template(
            "predict.html",
            prediction=prediction
        )

    except Exception as e:

        return render_template(
            "predict.html",
            error=str(e)
        )


@app.route("/dashboard")
def dashboard():

    metrics = {
        "dataset_size": 1338,
        "features": 8,
        "algorithm": "CatBoost Regressor",
        "problem": "Regression",
        "test_r2": "89%",
        "cv_r2": "83%"
    }

    models = [
        {"name": "Random Forest", "test": "84%", "cv": "80%"},
        {"name": "Gradient Boosting", "test": "89%", "cv": "83%"},
        {"name": "XGBoost", "test": "89%", "cv": "82%"},
        {"name": "CatBoost", "test": "89%", "cv": "83%"}
    ]

    return render_template(
        "dashboard.html",
        metrics=metrics,
        models=models
    )


@app.route("/about")
def about():
    return render_template("about.html")






        
        


if __name__ == "__main__":
    app.run(debug=True)
