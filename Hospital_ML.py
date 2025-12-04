import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder

Hospital_Data = pd.read_csv("....")
Hospital_Data.info()
Hospital_Data.head()
Hospital_Data.describe()
#--------------------
# feature engineering
#--------------------
Hospital_Data["addmission_date"] = pd.to_datetime(Hospital_Data["addmission_date"])
Hospital_Data["Year"] = Hospital_Data["addmission_date"].dt.year
Hospital_Data["Month"] = Hospital_Data["addmission_date"].dt.month
Hospital_Data["Day"] = Hospital_Data["addmission_date"].dt.day
Hospital_Data["Hour"] = Hospital_Data["addmission_date"].dt.hour
Hospital_Data["Total_Admissions"] = Hospital_Data["addmission_count"] + Hospital_Data["readmission_count"]
Hospital_Data = Hospital_Data.drop(columns=["addmission_date", "addmission_count", "readmission_count"])
Hospital_Data.to_csv("Clean_Hospital_Data.csv", index=False)

#-------------------------------------------------------
# preprocessing and defining a random forest pipeline
#-------------------------------------------------------
x = Hospital_Data.drop(columns=["Total_Admissions"])
y = Hospital_Data["Total_Admissions"]

cat_cols = x.select_dtypes(include="object").columns
num_cols = x.select_dtypes(include=np.number).columns

RF_preprocess = ColumnTransformer(
    transformers=[("cat", OrdinalEncoder(), cat_cols)],
    remainder="passthrough")

RF_pipeline = Pipeline([
    ("preprocess", RF_preprocess),
    ("model", RandomForestRegressor(n_estimators=500))])

Parameter_Distributions = {
    "model__max_depth": randint(5, 50),
    "model__min_samples_split": randint(2, 20)}
#-----------------------------------------
# Setting up and running cross validation RF
#----------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

RF_random_search = RandomizedSearchCV(
    estimator = RF_pipeline,
    param_distributions = Parameter_Distributions,
    n_iter= 20,
    cv= 5,
    scoring = "neg_mean_squared_error",
    n_jobs=-1
)

#-------------------------------------
# Making predictions on the test data RF
#-------------------------------------
RF_random_search.fit(x_train, y_train)
y_pred_RF = RF_random_search.predict(x_test)
print(f"Best model: {RF_random_search.best_estimator_}")
print(f"Best Parameters: {RF_random_search.best_params_}")

#-------------------------
# Testing model metrics RF
#-------------------------
rmse_RF = mean_squared_error(y_test, y_pred_RF, squared=False)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
r2_RF = r2_score(y_test, y_pred_RF)
print(f"RMSE: {rmse_RF:.3f}")
print(f"MAE: {mae_RF:.3f}")
print(f"R2: {r2_RF:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_RF, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual")
plt.savefig("Predicted_vs_Actual_RF.png", bbox_inches='tight', dpi=300)
plt.clf()

RF_residuals = y_test - y_pred_RF
plt.figure(figsize=(6,4))
plt.scatter(y_pred_RF, RF_residuals, color="green", alpha=0.7)
plt.hlines(0, y_pred_RF.min(), y_pred_RF.max(), linestyles="dashed", colors="red")
plt.xlabel("Predicted Total Admissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("Residual_Plot_RF.png", bbox_inches='tight', dpi=300)
plt.clf()

#-------
# Elastic Net model 
#------

#-------------------------------------------------------
# preprocessing and defining a EN pipeline
#-------------------------------------------------------
EN_preprocess = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

EN_pipeline = Pipeline([
    ("preprocess", EN_preprocess),
    ("model", ElasticNet(max_iter=10000))
])

#-----------------------------------------
# Setting up and running cross validation EN
#----------------------------------------
param_distributions = {
    "model__alpha": uniform(0.01, 10),
    "model__l1_ratio": uniform(0, 1)
}

EN_random_search = RandomizedSearchCV(
    estimator = EN_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#-------------------------------------
# Making predictions on the test data EN
#-------------------------------------
EN_random_search.fit(x_train, y_train)
y_pred_EN = EN_random_search.predict(x_test)
print(f"Best model: {EN_random_search.best_estimator_}")
print(f"Best Parameters: {EN_random_search.best_params_}")

#-------------------------
# Testing model metrics EN
#-------------------------
rmse_EN = mean_squared_error(y_test, y_pred_EN, squared=False)
mae_EN = mean_absolute_error(y_test, y_pred_EN)
r2_EN = r2_score(y_test, y_pred_EN)
print(f"RMSE: {rmse_EN:.3f}")
print(f"MAE: {mae_EN:.3f}")
print(f"R2: {r2_EN:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_EN, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual Elastic Net")
plt.savefig("Predicted_vs_Actual_EN.png", bbox_inches='tight', dpi=300)
plt.clf()

EN_residuals = y_test - y_pred_EN
plt.figure(figsize=(6,4))
plt.scatter(y_pred_EN, EN_residuals, color="green", alpha=0.7)
plt.hlines(0, y_pred_EN.min(), y_pred_EN.max(), linestyles="dashed", colors="red")
plt.xlabel("Predicted Total Admissions")
plt.ylabel("Residuals")
plt.title("Residual Plot Elastic Net")
plt.savefig("Residual_Plot_EN.png", bbox_inches='tight', dpi=300)
plt.clf()

#--------------------
# Comparison plot Predicted vs Actual for both models
#--------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_RF, color="blue", alpha=0.6, label="Random Forest")
plt.scatter(y_test, y_pred_EN, color="orange", alpha=0.6, label="Elastic Net")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual Comparison")
plt.legend()
plt.savefig("Predicted_vs_Actual_Comparison.png", bbox_inches='tight', dpi=300)
plt.show()

#--------------------
# Comparison metrics table
#--------------------
metrics = pd.DataFrame({
    "Model": ["Random Forest", "Elastic Net"],
    "RMSE": [rmse_RF, rmse_EN],
    "MAE": [mae_RF, mae_EN],
    "R2": [r2_RF, r2_EN]
})
print(metrics)
