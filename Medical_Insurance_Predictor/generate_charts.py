# import joblib
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load model
# model = joblib.load(r"C:\Users\ah266\OneDrive\Documents\Medical_Insurance_Predictor\models\catboost_model.pkl")

# # Feature names
# features = [
#     "Age",
#     "Sex",
#     "BMI",
#     "Children",
#     "Smoker",
#     "Northwest",
#     "Southeast",
#     "Southwest"
# ]

# importance = model.get_feature_importance()

# plt.figure(figsize=(8,5))

# plt.barh(features, importance)

# plt.xlabel("Importance")

# plt.title("CatBoost Feature Importance")

# plt.tight_layout()

# plt.savefig(r"C:\Users\ah266\OneDrive\Documents\Medical_Insurance_Predictor\static\charts\feature_importance.png")

# plt.close()

# print("Done")



from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import  matplotlib.pyplot as plt 
from catboost import CatBoostRegressor

df = pd.read_csv(r"C:\Users\ah266\OneDrive\Documents\machine learning\pandas practice\medical_insurance.csv")

df = df.drop_duplicates() 
df = pd.get_dummies(df , columns=['region'] , drop_first=True , dtype=np.int64) 
df['sex'] = df['sex'].map({'female': 0 , 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0 , 'yes': 1})

for col in df.iloc[: , [2]]:
    percentile25 = df[col].quantile(0.25)
    percentile75 = df[col].quantile(0.75)
    iqr = percentile75 - percentile25
    upper = percentile75 + 1.5*iqr
    lower = percentile25 - 1.5*iqr
    df[col] = np.where(
        df[col] > upper,
        upper,
        np.where(
            df[col] < lower ,
            lower ,
            df[col]
        )
    )

x = df.iloc[: , [0 , 1,2,3,4,6,7,8]]
y = df.iloc[: , -4]
y = np.log1p(y)

# Apply the SAME preprocessing you used while training
# (encoding, log transform, etc.)

# X = ...
# y = ...

# IMPORTANT:
# Use exactly the same preprocessing as your training notebook.

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2  ,  random_state=42)

ct = CatBoostRegressor(iterations=400,
    learning_rate=0.03,
    depth=4,
    random_state=42,
    verbose=0)
ct.fit(X_train , y_train)

pred = ct.predict(X_test)

pred = np.expm1(pred)
actual = np.expm1(y_test)

# plt.figure(figsize=(6,6))

# plt.scatter(actual, pred)

# plt.xlabel("Actual")

# plt.ylabel("Predicted")

# plt.title("Actual vs Predicted")

# plt.tight_layout()

# plt.savefig(r"charts\actual_vs_predicted.png")


# plt.close()


# residual = actual - pred

# plt.figure(figsize=(7,5))

# plt.scatter(pred, residual)

# plt.axhline(0, linestyle="--")

# plt.xlabel("Predicted")

# plt.ylabel("Residual")

# plt.title("Residual Plot")

# plt.tight_layout()

# plt.savefig(r"charts\residual.png")

# plt.close()


models = [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "CatBoost"
]

scores = [
    0.80,
    0.83,
    0.82,
    0.83
]

plt.figure(figsize=(7,5))

plt.bar(models, scores)

plt.ylim(.75,.90)

plt.ylabel("Cross Validation R²")

plt.title("Model Comparison")

plt.tight_layout()

plt.savefig(r"charts\model_comparison.png")

plt.close()