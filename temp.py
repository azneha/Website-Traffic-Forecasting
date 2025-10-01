import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

st.title("📈 Website Traffic Forecasting")

# Upload CSV or Excel
uploaded_file = st.file_uploader("Upload CSV or Excel with Date and Visitors", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.capitalize()  # ensures 'Date' and 'Visitors'
    
    # Convert Date to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
    st.write("### Raw Data", df.head())

    # Handle missing values
    df = df.asfreq('D').fillna(method='ffill')

    # Plot original data
    st.line_chart(df["Visitors"])

    # Model parameters
    st.sidebar.header("SARIMA Parameters")
    p = st.sidebar.number_input("AR (p)", min_value=0, max_value=10, value=2)
    d = st.sidebar.number_input("Difference (d)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("MA (q)", min_value=0, max_value=10, value=2)
    P = st.sidebar.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=1)
    D = st.sidebar.number_input("Seasonal Diff (D)", min_value=0, max_value=2, value=1)
    Q = st.sidebar.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=1)
    s = st.sidebar.number_input("Season Length (s)", min_value=1, max_value=365, value=7)
    steps = st.sidebar.number_input("Forecast Days", min_value=1, max_value=365, value=30)

    if st.button("Run Forecast"):
        # Fit SARIMA model
        model = SARIMAX(df["Visitors"],
                        order=(p,d,q),
                        seasonal_order=(P,D,Q,s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        st.write("### Model Summary")
        st.text(model_fit.summary())

        # Forecast
        forecast = model_fit.forecast(steps=int(steps))
        forecast_index = pd.date_range(df.index[-1] + timedelta(days=1), periods=int(steps), freq='D')
        forecast_df = pd.DataFrame({"Date": forecast_index, "Forecasted_Visitors": forecast.values})
        forecast_df.set_index("Date", inplace=True)
        st.write("### Forecast", forecast_df)

        # Plot forecast
        plt.figure(figsize=(12,6))
        plt.plot(df["Visitors"], label="Original")
        plt.plot(forecast_df["Forecasted_Visitors"], label="Forecast", color="red")
        plt.legend()
        plt.title("Website Traffic Forecast")
        plt.xlabel("Date")
        plt.ylabel("Visitors")
        st.pyplot(plt)
