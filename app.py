import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Config
st.set_page_config(page_title="Global Health Predictor", layout="wide")
st.title("🌍 Public Health Intelligence: Life Expectancy Predictor")

# 2. Load Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('longevity_model_v1.pkl')

art = load_artifacts()
model, scaler, feature_order, numeric_cols = art['model'], art['scaler'], art['feature_order'], art['numeric_cols']

# 3. User Input Layout
st.markdown("### National Health & Socio-Economic Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏥 Clinical & Mortality")
    adult_mort = st.number_input("Adult Mortality (per 1000)", 0, 1000, 150)
    hiv_aids = st.slider("HIV/AIDS Prevalence (%)", 0.0, 50.0, 0.1)
    under_five = st.number_input("Under-five Deaths (per 1000)", 0, 1000, 20)
    measles = st.number_input("Measles Cases (Reported)", 0, 100000, 100)
    bmi = st.slider("Average BMI", 10.0, 50.0, 25.0)
    thinness = st.slider("Thinness 1-19 years (%)", 0.0, 30.0, 5.0)

with col2:
    st.subheader("💉 Immunization & Health Ops")
    hepb = st.slider("Hepatitis B Coverage (%)", 0, 100, 80)
    polio = st.slider("Polio Coverage (%)", 0, 100, 80)
    diphtheria = st.slider("Diphtheria Coverage (%)", 0, 100, 80)
    alcohol = st.slider("Alcohol Consumption (L/capita)", 0.0, 20.0, 5.0)
    expenditure = st.slider("Total Health Expenditure (% GDP)", 0.0, 20.0, 5.0)

with col3:
    st.subheader("📈 Socio-Economic & Demographic")
    schooling = st.slider("Years of Schooling", 0, 20, 12)
    income_comp = st.slider("Income Composition of Resources (0-1)", 0.0, 1.0, 0.7)
    gdp_raw = st.number_input("GDP per Capita ($)", 100, 150000, 5000)
    pop_raw = st.number_input("Total Population", 1000, 2000000000, 10000000)
    perc_exp_raw = st.slider("Percentage Expenditure on Health (%)", 0, 100, 10)
    status = st.selectbox("Development Status", ["Developing", "Developed"])

# 4. Data Processing & Prediction
if st.button("Generate Prediction", type="primary"):
    # Convert 'status' to binary for calculation
    is_developing = 1 if status == "Developing" else 0
    
    # Create the raw input dictionary matching your feature_order names
    raw_data = {
        'Adult Mortality': adult_mort,
        'Alcohol': alcohol,
        'Hepatitis B': hepb,
        'Measles': measles,
        'BMI': bmi,
        'under-five deaths': under_five,
        'Polio': polio,
        'Total expenditure': expenditure,
        'Diphtheria': diphtheria,
        'HIV/AIDS': hiv_aids,
        'thinness  1-19 years': thinness,
        'Income composition of resources': income_comp,
        'Schooling': schooling,
        'log_GDP': np.log1p(gdp_raw),
        'log_Population': np.log1p(pop_raw),
        'log_percentage expenditure': np.log1p(perc_exp_raw),
        'Status_Schooling_Interaction': is_developing * schooling
    }
    
    # Convert to DF and reorder exactly to model's training order
    input_df = pd.DataFrame([raw_data])[feature_order]
    
    # Scale only the numeric columns identified in training
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Predict
    res = model.predict(input_df)[0]
    
    st.divider()
    st.balloons()
    st.metric(label="Estimated Life Expectancy", value=f"{res:.2f} Years")