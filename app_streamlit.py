import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from forecast import (
    Observation,
    NaiveForecast,
    SimpleAvgForecast,
    MovingAvgForecast,
    WeightedMovingAvgForecast,
    ExponentialSmoothingForecast,
    LinearForecast
)

# Page config
st.set_page_config(
    page_title="Forecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Global settings
DEFAULT_SETTINGS = {
    'forecast_horizon': 30,
    'confidence_interval': 0.95,
    'use_arima': True,
    'use_prophet': True,
    'use_lstm': True,
    'theme': 'light',
    'chart_style': 'default',
    'show_confidence': True,
    'show_legend': True,
    'show_grid': True
}

# Load settings
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_SETTINGS

# Save settings
def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

# Sidebar navigation
st.sidebar.title("Forecast Pro")
page = st.sidebar.radio("Navigation", ["Dashboard", "Analysis", "Settings"])

if page == "Dashboard":
    st.title("📊 Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{data['value'].mean():.2f}")
            with col2:
                st.metric("Standard Deviation", f"{data['value'].std():.2f}")
            with col3:
                st.metric("Count", len(data))
            
            # Time series plot
            st.subheader("Time Series Plot")
            fig = px.line(data, x='date', y='value', title='Demand Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "Analysis":
    st.title("📈 Analysis")
    
    # Mock data for demonstration
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'value': np.random.normal(100, 20, 100)
    })
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Demand", "150.5")
    with col2:
        st.metric("Standard Deviation", "25.3")
    with col3:
        st.metric("Seasonal Patterns", "1")
    with col4:
        st.metric("Outliers", "3")
    
    # Advanced visualizations
    st.subheader("Time Series Analysis")
    fig1 = px.line(data, x='date', y='value', title='Demand Over Time')
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution Analysis")
        fig2 = px.histogram(data, x='value', title='Demand Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Box Plot")
        fig3 = px.box(data, y='value', title='Demand Distribution Box Plot')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Model Performance
    st.subheader("Model Performance Comparison")
    model_results = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet', 'LSTM'],
        'MAE': [12.5, 13.2, 11.8],
        'RMSE': [15.3, 16.1, 14.9],
        'MAPE': [8.2, 8.7, 7.9],
        'R²': [0.85, 0.82, 0.87]
    })
    st.dataframe(model_results, use_container_width=True)

elif page == "Settings":
    st.title("⚙️ Settings")
    
    current_settings = load_settings()
    
    # Model Configuration
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            options=[7, 14, 30, 90],
            index=[7, 14, 30, 90].index(current_settings['forecast_horizon'])
        )
        
        confidence_interval = st.selectbox(
            "Confidence Interval",
            options=[0.8, 0.9, 0.95],
            index=[0.8, 0.9, 0.95].index(current_settings['confidence_interval'])
        )
    
    with col2:
        st.write("Model Selection")
        use_arima = st.checkbox("ARIMA", value=current_settings['use_arima'])
        use_prophet = st.checkbox("Prophet", value=current_settings['use_prophet'])
        use_lstm = st.checkbox("LSTM", value=current_settings['use_lstm'])
    
    # Display Settings
    st.subheader("Display Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            options=['light', 'dark', 'system'],
            index=['light', 'dark', 'system'].index(current_settings['theme'])
        )
        
        chart_style = st.selectbox(
            "Chart Style",
            options=['default', 'minimal', 'modern'],
            index=['default', 'minimal', 'modern'].index(current_settings['chart_style'])
        )
    
    with col2:
        st.write("Data Display")
        show_confidence = st.checkbox("Show Confidence Intervals", value=current_settings['show_confidence'])
        show_legend = st.checkbox("Show Chart Legend", value=current_settings['show_legend'])
        show_grid = st.checkbox("Show Grid Lines", value=current_settings['show_grid'])
    
    # Save settings
    if st.button("Save Settings"):
        new_settings = {
            'forecast_horizon': forecast_horizon,
            'confidence_interval': confidence_interval,
            'use_arima': use_arima,
            'use_prophet': use_prophet,
            'use_lstm': use_lstm,
            'theme': theme,
            'chart_style': chart_style,
            'show_confidence': show_confidence,
            'show_legend': show_legend,
            'show_grid': show_grid
        }
        save_settings(new_settings)
        st.success("Settings saved successfully!")
    
    # System Information
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Version: 1.0.0")
        st.write(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Data Points: 1000")
    
    with col2:
        st.write(f"Models Available: 3")
        st.write(f"Storage Used: 2.5 MB")
        st.write("System Status: Active") 