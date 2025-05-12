import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
import datetime

# ==== SETUP CONFIG + THEME ====
st.set_page_config(
    page_title="⏱️ Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== CSS CUSTOM ====
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: #fff !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(25,118,210,0.07);
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #1976d2 0%, #43e97b 100%);
        color: #fff;
        font-weight: 600;
        padding: 10px 18px;
        border: none;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(25,118,210,0.08);
        font-size: 1.1em;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #43e97b 0%, #1976d2 100%);
        color: #fff;
    }
    /* Blue sliders */
    [data-baseweb="slider"] .css-14g5kgc,
    [data-baseweb="slider"] .css-1gv0vcd,
    .stSlider > div > div > div[role="slider"] {
        background: #1976d2 !important;
        border-color: #1976d2 !important;
    }
    .stSlider .css-1c5b3bq {
        color: #1976d2 !important;
    }
    /* Blue checkboxes */
    .stCheckbox [data-baseweb="checkbox"] > div {
        border-color: #1976d2 !important;
    }
    .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] > div {
        background-color: #1976d2 !important;
        border-color: #1976d2 !important;
    }
    /* Card style for file uploader and controls */
    .block-container {
        max-width: 900px;
        margin: auto;
    }
    .stFileUploader, .stSelectbox, .stSlider, .stButton, .stDownloadButton {
        margin-bottom: 1.2em;
    }
    .card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(25,118,210,0.07);
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 2rem;
    }
    .section-title {
        color: #1976d2;
        font-size: 1.4em;
        font-weight: 700;
        margin-bottom: 1em;
        margin-top: 0.5em;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==== MAIN TITLE ====
st.markdown(
    "<h1 style='text-align: center; color:#1976d2; font-weight:700; letter-spacing:1px; margin-bottom:2rem;'>\U0001F4C8 Time Series Forecasting App</h1>",
    unsafe_allow_html=True
)

# ==== FILE UPLOAD ====
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("<b>Upload your CSV file</b> \\U0001F447", type=["csv"], key="fileuploader")
    st.markdown("</div>", unsafe_allow_html=True)
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>\U0001F4C4 Uploaded Data</div>", unsafe_allow_html=True)
        st.dataframe(data, height=250)
        st.markdown("</div>", unsafe_allow_html=True)

        # ==== TIME & VALUE COLUMN ====
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>\U0001F552 Select Columns</div>", unsafe_allow_html=True)
        time_column = st.selectbox("Select time column", data.columns, key="timecol")
        value_column_options = [col for col in data.columns if col != time_column]
        value_column = st.selectbox("Select value column", value_column_options, key="valcol")
        st.markdown("</div>", unsafe_allow_html=True)

        # Warn if user tries to select the same column (shouldn't be possible now)
        if time_column == value_column:
            st.error("Please select different columns for time and value.")
        else:
            # ==== FORECAST PERIOD ====
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>\U0001F52E Forecast Settings</div>", unsafe_allow_html=True)
            forecast_period = st.slider("Forecast steps", min_value=1, max_value=30, value=7, key="forecastslider")
            st.markdown("</div>", unsafe_allow_html=True)

            # ==== RUN FORECAST ====
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if st.button("\U0001F680 Forecast", key="forecastbtn"):
                try:
                    df = data[[time_column, value_column]].dropna()
                    df[time_column] = pd.to_datetime(df[time_column])
                    df = df.set_index(time_column)

                    model = ARIMA(df[value_column], order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=forecast_period)

                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_period)

                    forecast_df = pd.DataFrame({value_column: forecast}, index=future_dates)

                    # ==== CHART ====
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df[value_column], mode='lines', name='Actual', line=dict(color='#1976d2', width=3)))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[value_column], mode='lines+markers', name='Forecast', line=dict(color='#43e97b', width=3, dash='dash')))
                    fig.update_layout(
                        title="Forecast Chart",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template="plotly_white",
                        height=500,
                        font=dict(family='Segoe UI, Roboto, Arial, sans-serif', color='#222'),
                        plot_bgcolor='#fff',
                        paper_bgcolor='#fff',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ==== METRICS ====
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    import numpy as np
                    if len(df) > forecast_period:
                        true_values = df[value_column][-forecast_period:]
                        pred_values = forecast[:len(true_values)]
                        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
                        mae = mean_absolute_error(true_values, pred_values)
                        mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
                        st.markdown("### \U0001F4CA Model Evaluation")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"{rmse:.2f}")
                        col2.metric("MAE", f"{mae:.2f}")
                        col3.metric("MAPE", f"{mape:.2f}%")

                    # ==== EXPORT FORECAST ====
                    forecast_df_reset = forecast_df.reset_index()
                    forecast_df_reset.columns = ['Date', 'Forecast Value']
                    csv = forecast_df_reset.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Forecast Results", data=csv, file_name='forecast.csv', mime='text/csv')

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("⬆️ Please upload a CSV file with time series data to start forecasting.")
