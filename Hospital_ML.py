import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

#-------------------
# Loading in the data
#--------------------
Hospital_Data = pd.read_csv("Hospital_Dataset_2020_2024.csv")
print(Hospital_Data.info())
print(Hospital_Data.head())
print(Hospital_Data.describe())

#--------------------
# Feature engineering
#--------------------
Hospital_Data["Total_Admissions"] = Hospital_Data["admission_count"] + Hospital_Data["readmission_count"]
Hospital_Data["admission_date"] = pd.to_datetime(Hospital_Data["admission_date"])
Hospital_Data["Year"] = Hospital_Data["admission_date"].dt.year
Hospital_Data["Month"] = Hospital_Data["admission_date"].dt.month
Hospital_Data["Day"] = Hospital_Data["admission_date"].dt.day
Hospital_Data["Hour"] = Hospital_Data["admission_date"].dt.hour
Hospital_Data["DOW"] = Hospital_Data["admission_date"].dt.dayofweek
Hospital_Data["Lag_1"] = Hospital_Data["Total_Admissions"].shift(1)
Hospital_Data["Lag_24"] = Hospital_Data["Total_Admissions"].shift(24)
Hospital_Data["Rolling_3H"] = Hospital_Data["Total_Admissions"].rolling(3).mean()
Hospital_Data["Hour_DOW"] = Hospital_Data["Hour"] * Hospital_Data["DOW"]
Hospital_Data["Month_Season"] = Hospital_Data["Month"].astype(str) + "_" + Hospital_Data["seasonal_indicator"]
Hospital_Data = Hospital_Data.dropna().reset_index(drop=True)
Hospital_Data = Hospital_Data.drop(columns=["admission_date", "admission_count", "readmission_count"])
Hospital_Data.to_csv("Clean_Hospital_Data.csv", index=False)

#-------------------
# splitting up the data
#---------------------

x = Hospital_Data.drop(columns=["Total_Admissions"])
y = Hospital_Data["Total_Admissions"]

print("Features and response are split")

cat_cols = x.select_dtypes(include="object").columns
num_cols = x.select_dtypes(include=np.number).columns

train = Hospital_Data[Hospital_Data["Year"] < 2024]
test = Hospital_Data[Hospital_Data["Year"] == 2024]

x_train = train.drop(columns=["Total_Admissions"])
y_train = train["Total_Admissions"]
x_test = test.drop(columns=["Total_Admissions"])
y_test = test["Total_Admissions"]

print("Data is split")
#-------------------------------------------------------
# Preprocessing and defining a random forest pipeline
#-------------------------------------------------------

RF_preprocess = ColumnTransformer(
    transformers=[("cat", OrdinalEncoder(), cat_cols)],
    remainder="passthrough")

RF_pipeline = Pipeline([
    ("preprocess", RF_preprocess),
    ("model", RandomForestRegressor(n_estimators=500))])

print("Random Forest pipeline is made")

Parameter_Distributions = {
    "model__n_estimators": randint(300, 700),        
    "model__max_depth": randint(10, 50),            
    "model__min_samples_split": randint(2, 10),     
    "model__min_samples_leaf": randint(1, 5),       
    "model__max_features": ["sqrt", "log2", None]}

#-----------------------------------------
# Setting up and running cross validation RF
#----------------------------------------

RF_random_search = RandomizedSearchCV(
    estimator = RF_pipeline,
    param_distributions = Parameter_Distributions,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#-------------------------------------
# Making predictions on the test data RF
#-------------------------------------
print("Fitting Random Forest model...")

RF_random_search.fit(x_train, y_train)

print("RF CV is done")
print("Predicting with Random Forest model...")

y_pred_RF = RF_random_search.predict(x_test)

print(f"Best RF model: {RF_random_search.best_estimator_}")
print(f"Best RF Parameters: {RF_random_search.best_params_}")

#-------------------------
# Testing model metrics RF
#-------------------------
rmse_RF = root_mean_squared_error(y_test, y_pred_RF)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
r2_RF = r2_score(y_test, y_pred_RF)

print(f"Random Forest Metrics:")
print(f"RMSE: {rmse_RF:.3f}")
print(f"MAE: {mae_RF:.3f}")
print(f"R2: {r2_RF:.3f}")

RF_metrics = pd.DataFrame({
    "Model": ["Random Forest"],
    "RMSE": [rmse_RF],
    "MAE": [mae_RF],
    "R2": [r2_RF]})

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_RF, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual RF")
plt.savefig("Predicted_vs_Actual_RF.png", bbox_inches='tight', dpi=300)
plt.clf()

RF_residuals = y_test - y_pred_RF
plt.figure(figsize=(6,4))
plt.scatter(y_pred_RF, RF_residuals, color="green", alpha=0.7)
plt.hlines(0, y_pred_RF.min(), y_pred_RF.max(), linestyles="dashed", colors="red")
plt.xlabel("Predicted Total Admissions")
plt.ylabel("Residuals")
plt.title("Residual Plot RF")
plt.savefig("Residual_Plot_RF.png", bbox_inches='tight', dpi=300)
plt.clf()

#-------------------
# KNN pipeline
#-------------------
KNN_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

KNN_pipeline = Pipeline([
    ("preprocess", KNN_preprocessor),
    ("model", KNeighborsRegressor(n_jobs=-1))
])

#-------------------
# Hyperparameter tuning for KNN
#-------------------
KNN_parameter_distributions = {
    "model__n_neighbors": randint(5, 30),
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]
}

KNN_random_search = RandomizedSearchCV(
    estimator=KNN_pipeline,
    param_distributions=KNN_parameter_distributions,
    n_iter=50,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

#-------------------
# Fit and predict KNN
#-------------------
print("Fitting KNN model...")

KNN_random_search.fit(x_train, y_train)

print("KNN CV is done")
print("Predicting with KNN model...")

y_pred_KNN = KNN_random_search.predict(x_test)

print(f"Best KNN model: {KNN_random_search.best_estimator_}")
print(f"Best KNN Parameters: {KNN_random_search.best_params_}")

#-------------------
# Evaluate KNN
#-------------------
rmse_KNN = root_mean_squared_error(y_test, y_pred_KNN)
mae_KNN = mean_absolute_error(y_test, y_pred_KNN)
r2_KNN = r2_score(y_test, y_pred_KNN)
print(f"KNN Metrics:")
print(f"RMSE: {rmse_KNN:.3f}")
print(f"MAE: {mae_KNN:.3f}")
print(f"R2: {r2_KNN:.3f}")

#-------------------  
# Plot predictions KNN
#-------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_KNN, color="purple", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual KNN")
plt.savefig("Predicted_vs_Actual_KNN.png", bbox_inches='tight', dpi=300)
plt.clf()

# Residuals plot KNN
KNN_residuals = y_test - y_pred_KNN
plt.figure(figsize=(6,4))
plt.scatter(y_pred_KNN, KNN_residuals, color="orange", alpha=0.7)
plt.hlines(0, y_pred_KNN.min(), y_pred_KNN.max(), linestyles="dashed", colors="red")
plt.xlabel("Predicted Total Admissions")
plt.ylabel("Residuals")
plt.title("Residual Plot KNN")
plt.savefig("Residual_Plot_KNN.png", bbox_inches='tight', dpi=300)
plt.clf()

#----------------------------  
# Ensemble methods - Stacking
#----------------------------
stacked_model = StackingRegressor(
    estimators=[
        ("rf", RF_random_search.best_estimator_),
        ("knn", KNN_random_search.best_estimator_)
    ],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

#-------------------------------------
# Fit stacking model and making predictions
#-------------------------------------
print("Fitting stacked model (Random Forest + KNN)...")

stacked_model.fit(x_train, y_train)

print("Stacked model fitting done")
print("Predicting with stacked model...")

y_pred_stack = stacked_model.predict(x_test)

print("Prediction with stacked model complete")

#--------------------------------------
# Testing model metrics for Stacked Model
#--------------------------------------
rmse_stack = root_mean_squared_error(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)
r2_stack = r2_score(y_test, y_pred_stack)
print(f"Stacked Model Metrics:")
print(f"RMSE: {rmse_stack:.3f}")
print(f"MAE: {mae_stack:.3f}")
print(f"R2: {r2_stack:.3f}")

#--------------------
# Update comparison metrics table
#--------------------
stacked_metrics = pd.DataFrame({
    "Model": ["Random Forest", "KNN", "Stacked Model"],
    "RMSE": [rmse_RF, rmse_KNN, rmse_stack],
    "MAE": [mae_RF, mae_KNN, mae_stack],
    "R2": [r2_RF, r2_KNN, r2_stack]
})
print(stacked_metrics)
stacked_metrics.to_csv("stacked_metrecs.csv", index=False)

#--------------------
# Comparison plot Predicted vs Actual for all three models
#--------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_RF, color="blue", alpha=0.6, label="Random Forest")
plt.scatter(y_test, y_pred_KNN, color="orange", alpha=0.6, label="KNN")
plt.scatter(y_test, y_pred_stack, color="purple", alpha=0.6, label="Stacked Model")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Total Admissions")
plt.ylabel("Predicted Total Admissions")
plt.title("Predicted vs Actual Comparison - All Models")
plt.legend()
plt.savefig("Predicted_vs_Actual_All_Models.png", bbox_inches='tight', dpi=300)
