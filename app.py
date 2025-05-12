import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from forecast import (
    Observation,
    NaiveForecast,
    SimpleAvgForecast,
    MovingAvgForecast,
    WeightedMovingAvgForecast,
    ExponentialSmoothingForecast,
    LinearForecast,
    Forecast
)
from scipy import stats
from collections import Counter
from datetime import datetime
import json
import os

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global settings
DEFAULT_SETTINGS = {
    'forecast_horizon': 12,
    'confidence_interval': 0.95,
    'model_type': 'auto',
    'seasonality': 'additive',
    'theme': 'light',
    'chart_style': 'default',
    'show_confidence': True,
    'show_legend': True,
    'show_grid': True
}

def load_settings():
    """Load settings from JSON file"""
    try:
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to JSON file"""
    try:
        with open('settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        raise

def process_excel_data(df):
    """Process Excel data and return DataFrame"""
    try:
        if df is None or df.empty:
            raise ValueError("Empty or invalid Excel data")
            
        # Clean column names
        df.columns = [str(col).strip().lower() for col in df.columns]
        logger.debug(f"Cleaned columns: {df.columns.tolist()}")
        
        # Ensure we have date and value columns
        if 'date' not in df.columns or 'value' not in df.columns:
            raise ValueError("Excel file must contain 'date' and 'value' columns")
            
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing Excel data: {str(e)}")
        raise ValueError(f"Invalid Excel format: {str(e)}")

def main():
    st.set_page_config(
        page_title="Time Series Forecasting",
        page_icon="📈",
        layout="wide"
    )
    
    # Modern, clean, colorful interface
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background-color: #fff !important;
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        }
        .stMetric {
            background: #f8fafc;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 10px;
            margin-bottom: 10px;
        }
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, #1976d2 0%, #43e97b 100%);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1.2em;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(25,118,210,0.08);
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #43e97b 0%, #1976d2 100%);
            color: #fff;
        }
        .sidebar .sidebar-content {
            background: #f5f7fa;
        }
        /* Change slider color to blue */
        [data-baseweb="slider"] .css-14g5kgc,
        [data-baseweb="slider"] .css-1gv0vcd,
        .stSlider > div > div > div[role="slider"] {
            background: #1976d2 !important;
            border-color: #1976d2 !important;
        }
        .stSlider .css-1c5b3bq {
            color: #1976d2 !important;
        }
        /* Change checkbox color to blue */
        .stCheckbox [data-baseweb="checkbox"] > div {
            border-color: #1976d2 !important;
        }
        .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] > div {
            background-color: #1976d2 !important;
            border-color: #1976d2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='color:#1976d2; font-weight:700; letter-spacing:1px;'>Time Series Forecasting</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color:#1976d2;'>Settings</h2>", unsafe_allow_html=True)
    settings = load_settings()
    
    # Settings controls
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon",
        min_value=1,
        max_value=52,
        value=settings['forecast_horizon']
    )
    
    confidence_interval = st.sidebar.slider(
        "Confidence Interval",
        min_value=0.5,
        max_value=0.99,
        value=settings['confidence_interval'],
        step=0.01
    )
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        options=['auto', 'simple', 'holt', 'holt-winters'],
        index=['auto', 'simple', 'holt', 'holt-winters'].index(settings['model_type'])
    )
    
    seasonality = st.sidebar.selectbox(
        "Seasonality",
        options=['additive', 'multiplicative'],
        index=['additive', 'multiplicative'].index(settings['seasonality'])
    )
    
    theme = 'plotly_white'
    
    show_confidence = st.sidebar.checkbox(
        "Show Confidence Intervals",
        value=settings['show_confidence']
    )
    
    show_legend = st.sidebar.checkbox(
        "Show Legend",
        value=settings['show_legend']
    )
    
    show_grid = st.sidebar.checkbox(
        "Show Grid",
        value=settings['show_grid']
    )
    
    # Save settings
    new_settings = {
        'forecast_horizon': forecast_horizon,
        'confidence_interval': confidence_interval,
        'model_type': model_type,
        'seasonality': seasonality,
        'theme': 'light',
        'chart_style': settings['chart_style'],
        'show_confidence': show_confidence,
        'show_legend': show_legend,
        'show_grid': show_grid
    }
    save_settings(new_settings)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            df = process_excel_data(df)
            
            # Create forecast
            forecaster = Forecast(df, 'date', 'value')
            forecast_results = forecaster.generate_forecast(
                periods=forecast_horizon,
                seasonality=seasonality,
                confidence_level=confidence_interval,
                model_type=model_type
            )
            
            # Calculate metrics
            metrics = {
                'rmse': forecaster.calculate_rmse(),
                'mae': forecaster.calculate_mae(),
                'mape': forecaster.calculate_mape()
            }
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.2f}")
            with col3:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            # Create plot
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['value'],
                name='Historical',
                line=dict(color='#1976d2', width=3),
                marker=dict(color='#1976d2')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_results['date'],
                y=forecast_results['forecast'],
                name='Forecast',
                line=dict(color='#43e97b', width=3, dash='dash'),
                marker=dict(color='#43e97b')
            ))
            
            # Add confidence intervals
            if show_confidence:
                fig.add_trace(go.Scatter(
                    x=forecast_results['date'],
                    y=forecast_results['upper_bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,193,7,0.3)',
                    name='Upper Bound'
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_results['date'],
                    y=forecast_results['lower_bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,193,7,0.3)',
                    name='Lower Bound'
                ))
            
            # Update layout
            fig.update_layout(
                title='<b style="color:#1976d2;">Time Series Forecast</b>',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=show_legend,
                template=theme,
                plot_bgcolor='#fff',
                paper_bgcolor='#fff',
                font=dict(color='#222')
            )
            
            if show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button for forecast results
            csv = forecast_results.to_csv(index=False)
            st.download_button(
                label="Download Forecast Results",
                data=csv,
                file_name=f'forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error in main: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()