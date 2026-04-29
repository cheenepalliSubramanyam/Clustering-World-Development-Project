import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading saved files
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

df_country = pd.read_csv('final_country_data.csv',index_col='Country')
st.title("🌍 Country Development Cluster Predictor")

st.subheader("Option 1: Select a Country")
country = st.selectbox("Choose a country", df_country.index.tolist())

if st.button('Get Cluster and Development level'):
    level=df_country.loc[country,'Cluster_Name']
    st.write(f"{country} is classified as a {level} country")

st.subheader("Option 2: Enter Custom Values for Prediction")
st.write("Provide the development indicators below to predict the country's development cluster.")

# Inputs (kept in same order)
birth_rate = st.number_input("Birth Rate")
co2 = st.number_input("CO2 Emissions")
days = st.number_input("Days to Start Business")
energy = st.number_input("Energy Usage")
gdp = st.number_input("GDP")
health_exp = st.number_input("Health Exp % GDP")
health_capita = st.number_input("Health Exp/Capita")
infant = st.number_input("Infant Mortality Rate")
internet = st.number_input("Internet Usage")
lending = st.number_input("Lending Interest")
female_life = st.number_input("Life Expectancy Female")
male_life = st.number_input("Life Expectancy Male")
mobile = st.number_input("Mobile Phone Usage")
pop_0_14 = st.number_input("Population 0-14")
pop_15_64 = st.number_input("Population 15-64")
pop_65 = st.number_input("Population 65+")
pop_total = st.number_input("Population Total")
urban = st.number_input("Population Urban")
tour_in = st.number_input("Tourism Inbound")
tour_out = st.number_input("Tourism Outbound")

    
if st.button("Predict"):
    data = np.array([[birth_rate, co2, days, energy, gdp,
                      health_exp, health_capita, infant,
                      internet, lending, female_life, male_life,
                      mobile, pop_0_14, pop_15_64, pop_65,
                      pop_total, urban, tour_in, tour_out]])
    
    # data preprocessing
    data = np.log1p(data)
    data = scaler.transform(data)
    
    result = model.predict(data)
    
    st.markdown("### 📊 Prediction Result")

    cluster_map = {
        0: "Low Development Countries",
        1: "Moderate Development Countries",
        2: "High Development Countries",
        3: "Developing Countries"
    }

    predicted_cluster = result[0]
    cluster_name = cluster_map[predicted_cluster]

    st.success(f"🌍 The country belongs to Cluster {predicted_cluster}: **{cluster_name}**")