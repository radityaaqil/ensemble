import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.base import clone

# ========================= Load Dataset =========================
data = pd.read_csv('HousingData.csv')

# Drop rows with missing values
data = data.dropna()

# Features and Target
X = data.drop(columns=['MEDV'])  # Features (exclude target)
y = data['MEDV']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================= Define Base Models and Meta-Learner =========================
base_models = [
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42)),
    ('linear_regression', LinearRegression())
]

meta_learner = LinearRegression()

# ========================= Super Learner Implementation =========================
def super_learner(base_models, meta_learner, X_train, y_train, X_test, y_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store out-of-fold predictions for training meta-learner
    oof_predictions = np.zeros((X_train.shape[0], len(base_models)))
    test_predictions = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models):
        print(f"Training base model: {name}")
        fold_test_preds = np.zeros((X_test.shape[0], n_splits))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Clone model to ensure independence in folds
            model_clone = clone(model)
            model_clone.fit(X_tr, y_tr)
            
            # Out-of-fold predictions
            oof_predictions[val_idx, i] = model_clone.predict(X_val)
            
            # Predictions on test data
            fold_test_preds[:, fold] = model_clone.predict(X_test)
        
        # Average predictions for test data
        test_predictions[:, i] = fold_test_preds.mean(axis=1)
    
    # Train meta-learner on out-of-fold predictions
    print("\nTraining meta-learner...")
    meta_learner.fit(oof_predictions, y_train)
    
    # Final predictions using meta-learner
    final_predictions = meta_learner.predict(test_predictions)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, final_predictions)
    mse = mean_squared_error(y_test, final_predictions)
    r2 = r2_score(y_test, final_predictions)
    
    return mae, mse, r2

# ========================= Run Super Learner =========================
mae, mse, r2 = super_learner(base_models, meta_learner, X_train, y_train, X_test, y_test)

# Print evaluation metrics
print("\nSuper Learner Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
