import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Separate features (X) and target (y)
X = data.drop(columns=['MEDV'])  # MEDV is the target variable
y = data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ========================= Bagging =========================
# Baseline model
baseline_model = DecisionTreeRegressor(random_state=42)
baseline_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_baseline = baseline_model.predict(X_test)
r2_score_baseline = r2_score(y_test, y_pred_baseline)
print("Baseline Decision Tree:")
print("MAE:", mean_absolute_error(y_test, y_pred_baseline))
print("MSE:", mean_squared_error(y_test, y_pred_baseline))
print("R²:", r2_score_baseline)

# Bagging Regressor
max_depths = [1,3,5,8,10]
n_estimators = [50, 100, 200, 300, 500]
best_r2_bagging = -np.inf
best_params_bagging = {}
r2_results_bagging_regressor = np.zeros((len(max_depths), len(n_estimators)))

for i, depth in enumerate(max_depths):
    for j, estimator in enumerate(n_estimators):
        # Train Bagging Regressor with DecisionTreeRegressor
        bagging_model = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=depth, random_state=42),
            n_estimators=estimator,
            random_state=42,
        )
        bagging_model.fit(X_train, y_train)
        y_pred_bagging = bagging_model.predict(X_test)
        
        # Calculate R² and store it
        r2 = r2_score(y_test, y_pred_bagging)
        r2_results_bagging_regressor[i, j] = r2
        if r2 > best_r2_bagging:
            best_r2_bagging = r2
            best_params_bagging = {
                'n_estimators': estimator,
                'max_depth': depth
            }
        
        # Print metrics for each combination
        print(f"max_depth = {depth}, n_estimators = {estimator}")
        print("MAE:", mean_absolute_error(y_test, y_pred_bagging))
        print("MSE:", mean_squared_error(y_test, y_pred_bagging))
        print("R²:", r2)
        print("-" * 50)

# Random Forest Regressor
max_features = ['log2', 'sqrt', None]
best_r2_bagging_forest = -np.inf
best_params_bagging_forest = {}
r2_results_bagging_forest = np.zeros((len(max_features), len(n_estimators)))

for i, feature in enumerate(max_features):
    for j, estimator in enumerate(n_estimators):
        # Train Bagging Regressor with DecisionTreeRegressor
        bagging_model = BaggingRegressor(
            estimator=RandomForestRegressor(max_features=feature, random_state=42),
            n_estimators=estimator,
            random_state=42,
        )
        bagging_model.fit(X_train, y_train)
        y_pred_bagging = bagging_model.predict(X_test)
        
        # Calculate R² and store it
        r2 = r2_score(y_test, y_pred_bagging)
        r2_results_bagging_forest[i, j] = r2
        if r2 > best_r2_bagging_forest:
            best_r2_bagging_forest = r2
            best_params_bagging_forest = {
                'n_estimators': estimator,
                'max_features': feature
            }
        
        # Print metrics for each combination
        print(f"max_features = {feature}, n_estimators = {estimator}")
        print("MAE:", mean_absolute_error(y_test, y_pred_bagging))
        print("MSE:", mean_squared_error(y_test, y_pred_bagging))
        print("R²:", r2)
        print("-" * 50)

# ========================= GRAPH =========================
# Best R² from Bagging Regressor with Decision Tree
best_r2_bagging_tree = np.max(r2_results_bagging_regressor)

# Best R² from Bagging Regressor with Random Forest
best_r2_bagging_forest = np.max(r2_results_bagging_forest)

# R² from Baseline Decision Tree
r2_baseline = r2_score_baseline

# Compare models
models = ['Baseline Decision Tree', 'Bagging with Decision Tree', 'Bagging with Random Forest']
r2_scores = [r2_baseline, best_r2_bagging_tree, best_r2_bagging_forest]

# Plot Heatmap of R² Bagging Regressor
plt.figure(figsize=(10, 6))
sns.heatmap(r2_results_bagging_regressor, annot=True, cmap="viridis", xticklabels=n_estimators, yticklabels=max_depths)
plt.title("R² Score Heatmap for max_depth and n_estimators Bagging Regressor with Decision Tree Regressor")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.axhline(y=-0.5, color='red', linestyle='--', label=f'Baseline R²: {r2_score_baseline:.2f}')
plt.legend(loc='lower right')
plt.show()

# Plot Heatmap of R²
plt.figure(figsize=(10, 6))
sns.heatmap(r2_results_bagging_forest, annot=True, cmap="viridis", xticklabels=n_estimators, yticklabels=max_features)
plt.title("R² Score Heatmap for max_features and n_estimators Bagging Regressor with Random Forest Regressor")
plt.xlabel("Number of Estimators")
plt.ylabel("Max Features")
plt.axhline(y=-0.5, color='red', linestyle='--', label=f'Baseline R²: {r2_score_baseline:.2f}')
plt.legend(loc='lower right')
plt.show()

# Prepare data for comparison
models = ['Baseline', 'Bagging Regressor', 'Random Forest Regressor']
r2_scores = [r2_score_baseline, best_r2_bagging, best_r2_bagging_forest]

# Combine hyperparameters into readable strings
hyperparameters = [
    "No Params",
    f"n_estimators: {best_params_bagging['n_estimators']}\nmax_depth: {best_params_bagging['max_depth']}",
    f"n_estimators: {best_params_bagging_forest['n_estimators']}\nmax_features: {best_params_bagging_forest['max_features']}",
]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, r2_scores, color=['red', 'orange', 'green'])
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
