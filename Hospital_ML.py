import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

'''' RF example 
# ----------------------------------------------------------
# 1. Create fake dataset
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
# 5. Hyperparameter grid for tuning
# ----------------------------------------------------------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

# ----------------------------------------------------------
# 6. Train/test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ----------------------------------------------------------
# 7. GridSearchCV
# ----------------------------------------------------------
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ----------------------------------------------------------
# 8. Evaluate on test set
# ----------------------------------------------------------
preds = grid.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Test Predictions:", preds)
''''

