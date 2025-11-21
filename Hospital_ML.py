import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# clean data and feature engineer data (seperate dates and combine admission columns) 
# save the clean data
# build the RF pipeline 
# Split data into training and testing sets
# run CV

# Evaluate model performace metrics 
# Predicted vs actual (standard and over time) 
# Do the same for Elastic Net

'''' RF example 
# ----------------------------------------------------------
# 1. Fake dataset
# ----------------------------------------------------------
data = pd.DataFrame({
    "age": [25, 40, 60, 70, 33, 50, 45, 80],
    "severity": [1, 2, 3, 2, 1, 3, 2, 3],
    "hospital": ["A", "B", "A", "C", "C", "A", "B", "C"],
    "condition": ["asthma", "copd", "asthma", "heart", "asthma", "copd", "heart", "heart"],
    "length_of_stay": [3.1, 5.2, 4.8, 7.0, 2.9, 6.1, 5.5, 8.2]
})

X = data.drop(columns=["length_of_stay"])
y = data["length_of_stay"]

# ----------------------------------------------------------
# 2. Define column groups
# ----------------------------------------------------------
numeric_features = ["age", "severity"]
categorical_features = ["hospital", "condition"]

# ----------------------------------------------------------
# 3. Preprocessing
# ----------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ----------------------------------------------------------
# 4. Full pipeline
# ----------------------------------------------------------
pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", RandomForestRegressor(random_state=42))
])

# ----------------------------------------------------------
# 5. Randomized hyperparameter distributions
# ----------------------------------------------------------
param_dist = {
    "model__n_estimators": randint(100, 500),      # integer 100–500
    "model__max_depth": randint(5, 50),            # integer 5–50
    "model__min_samples_split": randint(2, 20)     # integer 2–20
}

# ----------------------------------------------------------
# 6. Train/test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------------------------------------
# 7. RandomizedSearchCV
# ----------------------------------------------------------
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=20,          # number of random combinations to try
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

# ----------------------------------------------------------
# 8. Evaluate on test set
# ----------------------------------------------------------
preds = random_search.predict(X_test)

print("Best Parameters:", random_search.best_params_)
print("Test Predictions:", preds)

# ----------------------------------------------------------
# Assuming 'random_search' is your fitted RandomizedSearchCV
# ----------------------------------------------------------
# Make predictions on the test set
y_pred = random_search.predict(X_test)

# ----------------------------------------------------------
# 9 Calculate regression metrics
# ----------------------------------------------------------
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# ----------------------------------------------------------
# 10 Predicted vs Actual scatter plot
# ----------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Length of Stay")
plt.ylabel("Predicted Length of Stay")
plt.title("Predicted vs Actual")
plt.show()

# ----------------------------------------------------------
# 11 Residual plot (errors vs predicted)
# ----------------------------------------------------------
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, color="green", alpha=0.7)
plt.hlines(0, y_pred.min(), y_pred.max(), linestyles="dashed", colors="red")
plt.xlabel("Predicted Length of Stay")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
''''


