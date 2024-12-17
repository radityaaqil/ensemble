from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ========================= Data Preparation =========================
# Load the dataset
data = pd.read_csv('HousingData.csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())
# Remove null values
data_cleaned = data.dropna()

# Separate features (X) and target (y)
X = data_cleaned.drop(columns=['MEDV'])  # MEDV is the target variable
y = data_cleaned['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================= Boosting =========================
# Baseline Model
baseline_model = DecisionTreeRegressor(random_state=42)
baseline_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_baseline = baseline_model.predict(X_test)
r2_score_baseline = r2_score(y_test, y_pred_baseline)
print("Baseline Decision Tree:")
print("MAE:", mean_absolute_error(y_test, y_pred_baseline))
print("MSE:", mean_squared_error(y_test, y_pred_baseline))
print("R²:", r2_score_baseline)

# Adaboost
learning_rates = [0.01, 0.1, 0.3, 0.5, 1]
n_estimators = [50, 100, 200, 300, 500]
best_r2_adaboost = -np.inf
best_params_adaboost = {}
r2_results_adaboost = np.zeros((len(learning_rates), len(n_estimators)))

for i, learning_rate in enumerate(learning_rates):
    for j, estimator in enumerate(n_estimators):
        # Train Adaboost Regressor with DecisionTreeRegressor
        adaboost_model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=estimator,
            learning_rate=learning_rate,
            random_state=42
        )
        adaboost_model.fit(X_train, y_train)
        y_pred_bagging = adaboost_model.predict(X_test)
        
        # Calculate R² and store it
        r2 = r2_score(y_test, y_pred_bagging)
        r2_results_adaboost[i, j] = r2
        if r2 > best_r2_adaboost:
            best_r2_adaboost = r2
            best_params_adaboost = {
                'n_estimators': estimator,
                'learning_rate': learning_rate
            }
        
        # Print metrics for each combination
        print(f"learning_rate = {learning_rate}, n_estimators = {estimator}")
        print("MAE:", mean_absolute_error(y_test, y_pred_bagging))
        print("MSE:", mean_squared_error(y_test, y_pred_bagging))
        print("R²:", r2)
        print("-" * 50)

# Gradient boosting
max_depths = [3, 5, 7]
best_r2_gb = -np.inf
best_params_gb = {}
r2_results_gradient_boosting = np.zeros((len(learning_rates), len(n_estimators), len(max_depths)))

for i, learning_rate in enumerate(learning_rates):
    for j, estimator in enumerate(n_estimators):
        for k, depth in enumerate(max_depths):
            # Train Gradient boosting with DecisionTreeRegressor
            gradient_boosting_model = GradientBoostingRegressor(
                n_estimators=estimator,
                learning_rate=learning_rate,
                max_depth=depth,
                random_state=42
            )
            gradient_boosting_model.fit(X_train, y_train)
            y_pred_bagging = gradient_boosting_model.predict(X_test)
            
            # Calculate R² and store it
            r2 = r2_score(y_test, y_pred_bagging)
            r2_results_gradient_boosting[i, j, k] = r2

            if r2 > best_r2_gb:
                best_r2_gb = r2
                best_params_gb = {
                    'n_estimators': estimator,
                    'learning_rate': learning_rate,
                    'max_depth': depth
                }
            
            # Print metrics for each combination
            print(f"learning_rate = {learning_rate}, n_estimators = {estimator}, max_depth = {depth}")
            print("MAE:", mean_absolute_error(y_test, y_pred_bagging))
            print("MSE:", mean_squared_error(y_test, y_pred_bagging))
            print("R²:", r2)
            print("-" * 50)

# XGBoost
subsamples = [0.5, 0.8, 1]
best_r2_xgb = -np.inf
best_params_xgb = {}
r2_results_xgboost = np.zeros((len(learning_rates), len(n_estimators), len(max_depths)))

for i, learning_rate in enumerate(learning_rates):
    for j, estimator in enumerate(n_estimators):
        for k, sample in enumerate(subsamples):
            # Train XGBoost Regressor with DecisionTreeRegressor
            gradient_boosting_model = XGBRegressor(
                n_estimators=estimator,
                learning_rate=learning_rate,
                subsample=sample,
                random_state=42
            )
            gradient_boosting_model.fit(X_train, y_train)
            y_pred_bagging = gradient_boosting_model.predict(X_test)
            
            # Calculate R² and store it
            r2 = r2_score(y_test, y_pred_bagging)
            r2_results_xgboost[i, j, k] = r2
            if r2 > best_r2_xgb:
                best_r2_xgb = r2
                best_params_xgb = {
                    'n_estimators': estimator,
                    'learning_rate': learning_rate,
                    'subsample': sample
                }
            
            # Print metrics for each combination
            print(f"learning_rate = {learning_rate}, n_estimators = {estimator}, subsample = {sample}")
            print("MAE:", mean_absolute_error(y_test, y_pred_bagging))
            print("MSE:", mean_squared_error(y_test, y_pred_bagging))
            print("R²:", r2)
            print("-" * 50)


plt.figure(figsize=(10, 6))
sns.heatmap(r2_results_adaboost, annot=True, cmap="viridis",
            xticklabels=n_estimators, yticklabels=learning_rates)
plt.title("AdaBoost R² Scores Heatmap")
plt.xlabel("Number of Estimators")
plt.ylabel("Learning Rate")
plt.axhline(y=-0.5, color='red', linestyle='--', label=f'Baseline R²: {r2_score_baseline:.2f}')
plt.legend(loc='lower right')
plt.show()

for idx, depth in enumerate(max_depths):
    plt.figure(figsize=(10, 6))
    sns.heatmap(r2_results_gradient_boosting[:, :, idx], annot=True, cmap="magma",
                xticklabels=n_estimators, yticklabels=learning_rates)
    plt.title(f"Gradient Boosting R² Scores Heatmap (Max Depth = {depth})")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Learning Rate")
    plt.axhline(y=-0.5, color='red', linestyle='--', label=f'Baseline R²: {r2_score_baseline:.2f}')
    plt.legend(loc='lower right')
    plt.show()

for idx, sample in enumerate(subsamples):
    plt.figure(figsize=(10, 6))
    sns.heatmap(r2_results_xgboost[:, :, idx], annot=True, cmap="coolwarm",
                xticklabels=n_estimators, yticklabels=learning_rates)
    plt.title(f"XGBoost R² Scores Heatmap (Subsample = {sample})")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Learning Rate")
    plt.axhline(y=-0.5, color='red', linestyle='--', label=f'Baseline R²: {r2_score_baseline:.2f}')
    plt.legend(loc='lower right')
    plt.show()

print("\nBest R² Score for AdaBoost:", best_r2_adaboost)
print("Best Hyperparameters:", best_params_adaboost)
print("\nBest R² Score for GradientBoosting:", best_r2_gb)
print("Best Hyperparameters:", best_params_gb)
print("\nBest R² Score for XGBoost:", best_params_xgb)
print("Best Hyperparameters:", best_params_xgb)

# Prepare data for comparison
models = ['Baseline', 'AdaBoost', 'Gradient Boosting', 'XGBoost']
r2_scores = [r2_score_baseline, best_r2_adaboost, best_r2_gb, best_r2_xgb]

# Combine hyperparameters into readable strings
hyperparameters = [
    "No Params",
    f"n_estimators: {best_params_adaboost['n_estimators']}\nlearning_rate: {best_params_adaboost['learning_rate']}",
    f"n_estimators: {best_params_gb['n_estimators']}\nlearning_rate: {best_params_gb['learning_rate']}\nmax_depth: {best_params_gb['max_depth']}",
    f"n_estimators: {best_params_xgb['n_estimators']}\nlearning_rate: {best_params_xgb['learning_rate']}\nsubsample: {best_params_xgb['subsample']}"
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, r2_scores, color=['red', 'orange', 'green', 'blue'])
plt.ylabel("R² Score")
plt.title("Comparison of Best R² Scores Between Models with Hyperparameters")

# Add annotations for R² scores and hyperparameters
for bar, r2, params in zip(bars, r2_scores, hyperparameters):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Centered horizontally
        bar.get_height() + 0.01,            # Slightly above the bar
        f"R²: {r2:.2f}\n{params}",          # Combine R² score and hyperparameters
        ha='center', va='bottom', fontsize=7
    )

# Adjust y-axis to make space for annotations
plt.ylim(0, max(r2_scores) + 0.1)
plt.xticks(rotation=15)
plt.show()