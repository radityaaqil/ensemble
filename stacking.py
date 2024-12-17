import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ========================= Load Dataset =========================
data = pd.read_csv('HousingData.csv')

# Drop rows with missing values
data = data.dropna()

# Features and Target
X = data.drop(columns=['MEDV'])  # Features (exclude target)
y = data['MEDV']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================= Define Base Models =========================
# Model Kombinasi 1
base_models_1 = [
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    # ('logistic_regression', LogisticRegression(max_iter=1000)),
    ('linear_regression', LinearRegression()),
    ('adaboost', AdaBoostRegressor(random_state=42))
]

# Model Kombinasi 2
base_models_2 = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svm', SVR(kernel='linear'))
]

# Model Kombinasi 3
base_models_3 = [
    ('adaboost', AdaBoostRegressor(random_state=42)),
    ('svm', SVR(kernel='linear')),
    ('decision_tree', DecisionTreeRegressor(random_state=42))
]

# Meta-Model
meta_model = LinearRegression()

# ========================= Define Stacking Function =========================
def stacking_experiment(base_models, meta_model, X_train, y_train, X_test, y_test, cv_folds=10):
    """
    Perform stacking with given base models and meta-model using cross-validation.
    """
    # Define Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=cv_folds)
    
    # Fit the model
    stacking_model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = stacking_model.predict(X_test)
    
    # Evaluate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mae, mse, r2

# ========================= Run Experiments =========================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Results for all combinations
print("===== Stacking Experiment Results =====\n")

# Kombinasi 1
print("Kombinasi 1: [Decision Tree, Logistic Regression, AdaBoost]")
mae1, mse1, r2_1 = stacking_experiment(base_models_1, meta_model, X_train, y_train, X_test, y_test, cv_folds=kf)
print(f"MAE: {mae1:.4f}, MSE: {mse1:.4f}, R²: {r2_1:.4f}\n")

# Kombinasi 2
print("Kombinasi 2: [Random Forest, Gradient Boosting, SVM]")
mae2, mse2, r2_2 = stacking_experiment(base_models_2, meta_model, X_train, y_train, X_test, y_test, cv_folds=kf)
print(f"MAE: {mae2:.4f}, MSE: {mse2:.4f}, R²: {r2_2:.4f}\n")

# Kombinasi 3
print("Kombinasi 3: [AdaBoost, SVM, Decision Tree]")
mae3, mse3, r2_3 = stacking_experiment(base_models_3, meta_model, X_train, y_train, X_test, y_test, cv_folds=kf)
print(f"MAE: {mae3:.4f}, MSE: {mse3:.4f}, R²: {r2_3:.4f}\n")

# ========================= Compare Results =========================
# Store results for plotting
models = ['Kombinasi 1', 'Kombinasi 2', 'Kombinasi 3']
r2_scores = [r2_1, r2_2, r2_3]

# Bar Plot for R² Comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(models, r2_scores, color=['skyblue', 'orange', 'green'])
plt.ylabel('R² Score')
plt.title('Comparison of R² Scores Between Stacking Model Combinations')
plt.ylim(0, 1)

# Add values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.4f}', ha='center', va='bottom')

plt.show()
