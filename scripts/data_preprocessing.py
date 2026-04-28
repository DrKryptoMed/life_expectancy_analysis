import pandas as pd
import numpy as np

def preprocess_data(filepath):
    """
    Cleans and prepares data, but leaves scaling for the training script 
    to prevent data leakage.
    """
    df = pd.read_csv(filepath)
    
    # 1. Column Sanitization
    df.columns = df.columns.str.strip()
    
    # 2. Validity Corrections
    df.loc[df['percentage expenditure'] > 100, 'percentage expenditure'] = np.nan
    df['Status'] = df['Status'].astype('category')
    
    # 3. Imputation Strategy
    mcar_cols = ['Life expectancy', 'Adult Mortality', 'Polio', 'Diphtheria']
    for col in mcar_cols:
        df[col] = df[col].fillna(df[col].median())
        
    country_mar_cols = ['Alcohol', 'BMI', 'thinness  1-19 years', 'thinness 5-9 years']
    for col in country_mar_cols:
        df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.median()))
        
    df['Population'] = df.groupby('Country')['Population'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    ).fillna(df['Population'].median())
    
    # 4. Feature Engineering
    cols_to_drop = ['Country', 'Year', 'infant deaths', 'thinness 5-9 years']
    df_eng = df.drop(columns=cols_to_drop)
    
    # Log Transformation
    skewed_cols = ['GDP', 'Population', 'percentage expenditure']
    for col in skewed_cols:
        if col in df_eng.columns:
            df_eng[f'log_{col}'] = np.log1p(df_eng[col])
    df_eng = df_eng.drop(columns=skewed_cols, errors='ignore')
    
    # Encoding
    df_eng = pd.get_dummies(df_eng, columns=['Status'], drop_first=True)
    
    # Interaction Term (Calculated before scaling for logic)
    df_eng['Status_Schooling_Interaction'] = df_eng['Status_Developing'] * df_eng['Schooling']
    
    return df_eng

if __name__ == "__main__":
    data = preprocess_data('./data/Life Expectancy Data.csv')
    print(f"Features prepared: {data.columns.tolist()}")