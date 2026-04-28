import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def train_and_export_model(df):
    # 1. Feature/Target Split
    X = df.drop(columns=['Life expectancy'])
    y = df['Life expectancy']
    
    # 2. Train-Test Split (Problem 1 Resolved: Split happens FIRST)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Dynamic Column Identification (Problem 3 Resolved: Interaction logic)
    # We identify numeric columns, excluding binary indicators
    binary_cols = ['Status_Developing']
    numeric_cols = [col for col in X_train.columns if col not in binary_cols]
    
    # 4. Fit Scaler ONLY on Training Data (Problem 2 & 5 Resolved)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 5. Fix Interaction Inconsistency (Problem 3 Resolved)
    # We re-calculate interaction AFTER scaling the parent features
    X_train_scaled['Status_Schooling_Interaction'] = X_train_scaled['Status_Developing'] * X_train_scaled['Schooling']
    X_test_scaled['Status_Schooling_Interaction'] = X_test_scaled['Status_Developing'] * X_test_scaled['Schooling']

    # 6. Training with Explicit Feature Ordering (Problem 4 Resolved)
    # We ensure the column order is locked before fitting
    feature_order = X_train_scaled.columns.tolist()
    model = LinearRegression()
    model.fit(X_train_scaled[feature_order], y_train)
    
    # 7. Evaluation
    y_pred = model.predict(X_test_scaled[feature_order])
    print(f"Deployment Metrics -> R2: {r2_score(y_test, y_pred):.4f}, MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    # 8. Exporting Artifacts (Problem 5 Resolved)
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_order': feature_order,
        'numeric_cols': numeric_cols
    }
    joblib.dump(artifacts, 'longevity_model_v1.pkl')
    print("Model Artifacts exported successfully.")

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    # Load and Preprocess
    df_ready = preprocess_data('./data/Life Expectancy Data.csv')
    train_and_export_model(df_ready)