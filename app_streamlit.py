import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from forecast import Forecast

# Page config
st.set_page_config(
    page_title="Demand Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("Demand Forecast Application")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display the data
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Date column selection
        date_col = st.selectbox("Select Date Column", df.columns)
        
        # Value column selection
        value_col = st.selectbox("Select Value Column", df.columns)

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)

        # Forecast parameters
        st.subheader("Forecast Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_periods = st.number_input("Forecast Periods", min_value=1, value=12)
            seasonality = st.selectbox("Seasonality", ["Additive", "Multiplicative"])
        
        with col2:
            confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
            model_type = st.selectbox("Model Type", ["Auto", "Simple", "Holt", "Holt-Winters"])

        if st.button("Generate Forecast"):
            # Create forecast object
            forecast = Forecast(df, date_col, value_col)
            
            # Generate forecast
            forecast_df = forecast.generate_forecast(
                periods=forecast_periods,
                seasonality=seasonality.lower(),
                confidence_level=confidence_level,
                model_type=model_type.lower()
            )

            # Plot the results
            st.subheader("Forecast Results")
            
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[value_col],
                name="Historical",
                line=dict(color="blue")
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                name="Forecast",
                line=dict(color="red")
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                name="Upper Bound",
                line=dict(color="red", dash="dash"),
                opacity=0.3
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                name="Lower Bound",
                line=dict(color="red", dash="dash"),
                opacity=0.3,
                fill="tonexty"
            ))
            
            fig.update_layout(
                title="Forecast Results",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast metrics
            st.subheader("Forecast Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'MAPE'],
                'Value': [
                    forecast.calculate_rmse(),
                    forecast.calculate_mae(),
                    forecast.calculate_mape()
                ]
            })
            st.dataframe(metrics_df)
            
            # Download forecast results
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Forecast Results",
                data=csv,
                file_name="forecast_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV or Excel file to begin.") 