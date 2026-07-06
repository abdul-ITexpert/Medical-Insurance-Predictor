# 🏥 AI-Powered Medical Insurance Cost Prediction

A Machine Learning web application built with **Python, Flask, and CatBoost** that predicts an individual's annual medical insurance cost based on demographic and health-related information.

---

## 📌 Project Overview

Medical insurance companies estimate insurance premiums using several factors such as age, BMI, smoking status, and family size. This project automates that process by using a trained **CatBoost Regressor** to predict insurance costs accurately through an interactive web application.

The application allows users to enter their information and instantly receive an estimated insurance cost.

---

## 🚀 Features

* Predict annual medical insurance cost
* User-friendly Flask web interface
* Machine Learning dashboard
* Model performance comparison
* Feature importance visualization
* Responsive Bootstrap design
* Error handling and input validation

---

## 📊 Dataset

The project uses the **Medical Insurance Cost Prediction Dataset**.

### Dataset Information

* Total Records: **1338**
* Input Features: **8**
* Target Variable: **Insurance Charges**

### Features Used

* Age
* Sex
* BMI
* Children
* Smoker
* Region (One-Hot Encoded)

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

* Checked for missing values
* Removed duplicate records
* Removed outliers
* One-hot encoding for the Region feature
* Binary encoding for Sex and Smoker
* Log transformation (`log1p`) applied to the target variable (`charges`) to improve model performance

---

## 🤖 Machine Learning Models Evaluated

Several regression models were trained and compared.

| Model             | Test R² | Cross Validation R² |
| ----------------- | ------- | ------------------- |
| Random Forest     | 84%     | 80%                 |
| Gradient Boosting | 89%     | 83%                 |
| XGBoost           | 89%     | 82%                 |
| CatBoost          | 89%     | 83%                 |

### Selected Model

**CatBoost Regressor**

CatBoost was selected because it achieved one of the highest prediction accuracies while maintaining strong cross-validation performance, demonstrating good generalization.

---

## 💻 Technologies Used

### Programming Languages

* Python
* HTML5
* CSS3
* JavaScript

### Frameworks

* Flask
* Bootstrap 5

### Machine Learning Libraries

* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Joblib

### Visualization

* Matplotlib

---

## 📂 Project Structure

```text
Medical_Insurance_Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── catboost_model.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── charts/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   └── about.html
│
└── notebook/
    └── Insurance_Cost_Prediction.ipynb
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/abdul-ITexpert/Medical-Insurance-Predictor.git 
```

### Navigate to the project

```bash
cd Medical-Insurance-Predictor
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🧪 Model Workflow

```
Dataset
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
    ↓
CatBoost Model Selection
    ↓
Model Saving
    ↓
Flask Web Application
    ↓
Insurance Cost Prediction
```

---

## 📈 Performance

* Test R² Score: **89%**
* Cross Validation R²: **83%**
* Target Transformation: **log1p**
* Prediction Inverse Transformation: **expm1**

---

## 🌐 Web Application

The application includes:

* Home Page
* Insurance Cost Prediction Page
* Dashboard
* About Page

Users can enter their information and receive an estimated insurance cost in real time.

---

## 🔮 Future Improvements

* User authentication
* Prediction history
* Interactive dashboards
* Cloud database integration
* API support
* Mobile application
* Real-time insurance recommendations

---

## 👨‍💻 Author

**Abdul Hanan**

Bachelor of Science in Information Technology (BSIT)

Machine Learning | Artificial Intelligence | AI Automation | Wordpress Developer

---

## 📄 License

This project is developed for educational and academic purposes.
