import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import datetime

st.set_page_config(
    page_title="⏱️ Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data, height=250)

    time_column = st.selectbox("Select time column", data.columns)
    value_column = st.selectbox("Select value column", data.columns)
    forecast_period = st.slider("Forecast steps", min_value=1, max_value=30, value=7)

    if st.button("Forecast"):
        try:
            if time_column == value_column:
                st.error("Please select different columns for time and value.")
            else:
                df = data[[time_column, value_column]].dropna()
                df[time_column] = pd.to_datetime(df[time_column])
                df = df.set_index(time_column)

                model = ARIMA(df[value_column], order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=forecast_period)

                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_period)

                forecast_df = pd.DataFrame({value_column: forecast}, index=future_dates)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines', name='Actual'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[value_column], mode='lines+markers', name='Forecast'))
                fig.update_layout(
                    title="Forecast Chart",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                from sklearn.metrics import mean_squared_error, mean_absolute_error
                import numpy as np
                if len(df) > forecast_period:
                    true_values = df[value_column][-forecast_period:]
                    pred_values = forecast[:len(true_values)]
                    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
                    mae = mean_absolute_error(true_values, pred_values)
                    mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
                    st.markdown("### Model Evaluation")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{rmse:.2f}")
                    col2.metric("MAE", f"{mae:.2f}")
                    col3.metric("MAPE", f"{mape:.2f}%")

                forecast_df_reset = forecast_df.reset_index()
                forecast_df_reset.columns = ['Date', 'Forecast Value']
                csv = forecast_df_reset.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast Results", data=csv, file_name='forecast.csv', mime='text/csv')
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file with time series data to start forecasting.")
