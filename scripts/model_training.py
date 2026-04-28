import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def train_longevity_model(df):
    """
    Performs splitting, scaling, and training. 
    Exports the final artifacts for deployment.
    """
    # 1. Defining Features (X) and Target (y)
    X = df.select_dtypes(include=[np.number]).drop(columns=['Life expectancy'], errors='ignore')
    y = df['Life expectancy']

    # 2. Sequential Splitting (80% Train/Val, 20% Final Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Further split Train into Train (90%) and Val (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

    # 3. Targeted Scaling (Preventing Data Leakage)
    binary_cols = ['Status_Developing']
    cols_to_scale = [col for col in X_train.columns if col not in binary_cols]
    
    scaler = StandardScaler()
    
    # Fit ONLY on X_train
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    # Transform Val and Test using Train parameters
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 4. Re-calculate Interaction after Scaling
    for dataset in [X_train, X_val, X_test]:
        dataset['Status_Schooling_Interaction'] = dataset['Status_Developing'] * dataset['Schooling']

    # 5. Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    def report_metrics(name, y_true, y_pred):
        print(f"--- {name} Results ---")
        print(f"R2: {r2_score(y_true, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}\n")

    report_metrics("Validation Set", y_val, model.predict(X_val))
    report_metrics("Final Test Set", y_test, model.predict(X_test))

    # 7. Artifact Export
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_order': X_train.columns.tolist(),
        'numeric_cols': cols_to_scale
    }
    joblib.dump(artifacts, 'longevity_model_v1.pkl')
    print("✅ Model Artifact 'longevity_model_v1.pkl' exported for deployment.")

if __name__ == "__main__":
    from data_preprocessing import clean_and_engineer
    # Integration logic
    PATH = '../data/Life Expectancy Data.csv'
    df_engineered = clean_and_engineer(pd.read_csv(PATH))
    train_longevity_model(df_engineered)