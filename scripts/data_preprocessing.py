import pandas as pd
import numpy as np

def clean_and_engineer(df):
    """
    Performs data cleaning, imputation, and feature engineering.
    Prepares the 'df_engineered' dataset for statistical modeling.
    """
    # 1. Column Sanitization
    df.columns = df.columns.str.strip()
    df['Status'] = df['Status'].astype('category')
    
    # 2. Validity Correction
    # Correcting boundary error for health expenditure
    df.loc[df['percentage expenditure'] > 100, 'percentage expenditure'] = np.nan
    
    # 3. Imputation Strategy
    # MCAR: Global Median
    mcar_cols = ['Life expectancy', 'Adult Mortality', 'Polio', 'Diphtheria']
    for col in mcar_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # MAR: Grouped Medians (Biological/Cultural)
    country_mar_cols = ['Alcohol', 'BMI', 'thinness  1-19 years', 'thinness 5-9 years']
    for col in country_mar_cols:
        df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.median()))
        
    # Infrastructure/Policy: Interpolation & Status Median Fallback
    df['Population'] = df.groupby('Country')['Population'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    ).fillna(df.groupby('Status')['Population'].transform('median'))
    
    # Remaining missing values handled by global median
    df = df.fillna(df.median(numeric_only=True))

    # 4. Feature Engineering
    # Resolving Multicollinearity
    cols_to_drop = ['Country', 'Year', 'infant deaths', 'thinness 5-9 years']
    df_eng = df.drop(columns=cols_to_drop)
    
    # Normalizing Skewed Data (Log Transformation)
    skewed_cols = ['GDP', 'Population', 'percentage expenditure']
    for col in skewed_cols:
        if col in df_eng.columns:
            df_eng[f'log_{col}'] = np.log1p(df_eng[col])
    df_eng = df_eng.drop(columns=skewed_cols)
    
    # Encoding Categorical Status
    df_eng = pd.get_dummies(df_eng, columns=['Status'], drop_first=True)
    
    # Interaction Term (Preliminary)
    df_eng['Status_Schooling_Interaction'] = df_eng['Status_Developing'] * df_eng['Schooling']
    
    return df_eng

if __name__ == "__main__":
    # Example usage for local testing
    PATH = '../data/Life Expectancy Data.csv'
    raw_df = pd.read_csv(PATH)
    processed_df = clean_and_engineer(raw_df)
    print(f"Preprocessing Complete. Features: {processed_df.shape[1]}")