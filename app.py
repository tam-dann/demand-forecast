import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash, jsonify, send_file
import traceback
import logging
import io
from forecast import (
    Observation,
    NaiveForecast,
    SimpleAvgForecast,
    MovingAvgForecast,
    WeightedMovingAvgForecast,
    ExponentialSmoothingForecast,
    LinearForecast
)
from scipy import stats
from collections import Counter
from io import StringIO
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Load settings from file if exists
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_SETTINGS

# Save settings to file
def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def process_input_data(values):
    """Process input data and return list of Observations"""
    try:
        logger.debug(f"Processing input values: {values}")
        # Convert all values to float and filter out empty values
        processed_values = []
        for val in values:
            if val and str(val).strip():
                try:
                    processed_values.append(float(val))
                except ValueError:
                    logger.warning(f"Skipping invalid value: {val}")
        
        if not processed_values:
            raise ValueError("No valid numbers provided")
        
        logger.debug(f"Processed values: {processed_values}")
        return [Observation(i + 1, val) for i, val in enumerate(processed_values)]
    except Exception as e:
        logger.error(f"Error processing values: {str(e)}")
        raise ValueError(f"Invalid data format: {str(e)}")

def process_csv_input(csv_text):
    """Process CSV input text and return list of Observations"""
    try:
        if not csv_text or not csv_text.strip():
            raise ValueError("Empty CSV input")
            
        # Split by comma and clean up each value
        values = [val.strip() for val in csv_text.split(',')]
        # Filter out empty values and convert to float
        processed_values = []
        for val in values:
            if val:
                try:
                    processed_values.append(float(val))
                except ValueError:
                    logger.warning(f"Skipping invalid value: {val}")
        
        if not processed_values:
            raise ValueError("No valid numbers provided")
            
        return [Observation(i + 1, val) for i, val in enumerate(processed_values)]
    except Exception as e:
        logger.error(f"Error processing CSV input: {str(e)}")
        raise ValueError(f"Invalid CSV format: {str(e)}")

def process_excel_data(df):
    """Process Excel data and return list of Observations"""
    try:
        if df is None or df.empty:
            raise ValueError("Empty or invalid Excel data")
            
        # Clean column names
        df.columns = [str(col).strip().lower() for col in df.columns]
        logger.debug(f"Cleaned columns: {df.columns.tolist()}")
        
        # Check if the DataFrame has the expected columns
        if 'period' in df.columns and 'demand' in df.columns:
            # Convert to numeric, handling any non-numeric values
            df['period'] = pd.to_numeric(df['period'], errors='coerce')
            df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
            
            # Drop rows with NaN values
            df = df.dropna(subset=['period', 'demand'])
            
            if df.empty:
                raise ValueError("No valid data rows found after processing")
            
            # Sort by period
            df = df.sort_values('period')
            
            # Check for negative values
            if (df['demand'] < 0).any():
                raise ValueError("Demand values cannot be negative")
            
            # Check for non-sequential periods
            periods = df['period'].tolist()
            if not all(periods[i] <= periods[i+1] for i in range(len(periods)-1)):
                raise ValueError("Periods must be in sequential order")
            
            # Create observations
            observations = []
            for _, row in df.iterrows():
                try:
                    period = float(row['period'])
                    demand = float(row['demand'])
                    observations.append(Observation(period, demand))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid row: {row}, error: {str(e)}")
            
            if not observations:
                raise ValueError("No valid data values found")
                
            logger.debug(f"Successfully processed {len(observations)} observations")
            return observations
        else:
            raise ValueError("File must contain 'period' and 'demand' columns")
            
    except Exception as e:
        logger.error(f"Error processing Excel data: {str(e)}")
        raise ValueError(f"Error processing Excel file: {str(e)}")

def calculate_statistics(data):
    """Calculate statistical measures for the input data"""
    try:
        if not data:
            logger.warning("No data provided for statistics calculation")
            return None

        values = [obs.demand for obs in data if obs.demand is not None]
        if not values:
            logger.warning("No valid demand values for statistics calculation")
            return None

        # Calculate basic statistics
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values))
        variance = float(np.var(values))
        range_val = float(np.max(values) - np.min(values))
        
        # Calculate mode safely
        try:
            mode_result = stats.mode(values, keepdims=False)
            mode_value = float(mode_result.mode) if hasattr(mode_result, 'mode') else mode_result[0]
        except Exception as e:
            logger.warning(f"Error calculating mode: {str(e)}")
            # If mode calculation fails, use the most common value
            mode_value = float(Counter(values).most_common(1)[0][0])
        
        # Calculate trend
        try:
            periods = [obs.period for obs in data if obs.period is not None]
            if len(periods) >= 2:
                trend = float(np.polyfit(periods, values, 1)[0])
            else:
                trend = 0.0
        except Exception as e:
            logger.warning(f"Error calculating trend: {str(e)}")
            trend = 0.0
        
        return {
            'mean': mean,
            'median': median,
            'mode': mode_value,
            'std': std,
            'variance': variance,
            'range': range_val,
            'trend': trend
        }
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def detect_seasonality(data, max_period=12):
    """Detect seasonality in the data using autocorrelation"""
    values = [obs.demand for obs in data]
    if len(values) < 4:  # Need at least 4 points for meaningful seasonality detection
        return None
    
    # Calculate autocorrelation
    acf = np.correlate(values, values, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]  # Normalize
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(1, min(len(acf)-1, max_period)):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.5:
            peaks.append(i)
    
    return peaks if peaks else None

def calculate_metrics(data, forecasts):
    """Calculate additional performance metrics for the model"""
    try:
        actual = [obs.demand for obs in data if obs.demand is not None]
        predicted = [obs.forecast for obs in forecasts if obs.forecast is not None]
        
        if not actual or not predicted:
            return {
                'rmse': None,
                'mape': None,
                'r2': None
            }
        
        # Ensure we only compare the same number of points
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))
        
        # Calculate MAPE
        mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
        
        # Calculate R²
        ss_tot = np.sum((np.array(actual) - np.mean(actual)) ** 2)
        ss_res = np.sum((np.array(actual) - np.array(predicted)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'rmse': None,
            'mape': None,
            'r2': None
        }

def detect_outliers(data, threshold=2):
    """Detect outliers in the data using z-score method"""
    values = [obs.demand for obs in data if obs.demand is not None]
    if not values:
        return []
    
    z_scores = np.abs((values - np.mean(values)) / np.std(values))
    outliers = [i for i, z in enumerate(z_scores) if z > threshold]
    return outliers

def create_advanced_charts(data, stats):
    """Create multiple visualization charts with enhanced features"""
    try:
        if not data:
            logger.warning("No data provided for charts")
            return {
                'time_series': '',
                'distribution': '',
                'box_plot': ''
            }

        # Extract values and periods, filtering out None values
        values = [obs.demand for obs in data if obs.demand is not None]
        periods = [obs.period for obs in data if obs.period is not None]
        
        if not values or not periods:
            logger.warning("No valid values or periods for charts")
            return {
                'time_series': '',
                'distribution': '',
                'box_plot': ''
            }
        
        logger.debug(f"Creating charts with {len(values)} data points")
        
        # Main time series plot
        fig1 = go.Figure()
        
        # Add actual values
        fig1.add_trace(go.Scatter(
            x=periods,
            y=values,
            mode='lines+markers',
            name='Demand',
            line=dict(color='#3498db'),
            marker=dict(size=8)
        ))
        
        # Add trend line if we have enough points
        if len(values) >= 2:
            try:
                # Convert to numpy arrays for calculation
                x = np.array(periods, dtype=float)
                y = np.array(values, dtype=float)
                
                # Calculate trend
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Generate trend points for the same x values
                trend_values = p(x)
                
                fig1.add_trace(go.Scatter(
                    x=x.tolist(),
                    y=trend_values.tolist(),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='#2c3e50')
                ))
            except Exception as e:
                logger.warning(f"Error calculating trend: {str(e)}")
        
        fig1.update_layout(
            title='Demand Over Time',
            xaxis_title='Period',
            yaxis_title='Demand',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        # Distribution plot
        try:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=values,
                name='Demand Distribution',
                marker_color='#3498db'
            ))
            fig2.update_layout(
                title='Demand Distribution',
                xaxis_title='Demand',
                yaxis_title='Frequency',
                template='plotly_white'
            )
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            fig2 = go.Figure()
        
        # Box plot
        try:
            fig3 = go.Figure()
            fig3.add_trace(go.Box(
                y=values,
                name='Demand',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig3.update_layout(
                title='Demand Distribution Box Plot',
                yaxis_title='Demand',
                template='plotly_white'
            )
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            fig3 = go.Figure()
        
        # Convert figures to HTML
        charts = {
            'time_series': fig1.to_html(full_html=False, include_plotlyjs='cdn'),
            'distribution': fig2.to_html(full_html=False, include_plotlyjs='cdn'),
            'box_plot': fig3.to_html(full_html=False, include_plotlyjs='cdn')
        }
        
        logger.debug("Charts created successfully")
        return charts
        
    except Exception as e:
        logger.error(f"Error creating charts: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'time_series': '',
            'distribution': '',
            'box_plot': ''
        }

def compare_models(data, forecast_steps=3):
    """Compare multiple forecasting models with different parameters"""
    try:
        if not data or len(data) < 2:
            logger.warning("Not enough data for model comparison")
            return []

        logger.debug(f"Starting model comparison with {len(data)} data points")
        logger.debug(f"Data points: {[(obs.period, obs.demand) for obs in data]}")

        # Create a copy of the data to avoid modifying the original
        data_copy = [Observation(obs.period, obs.demand) for obs in data]
        
        # Initialize models based on data length
        models = {
            "Naive": NaiveForecast(),
            "Simple Average": SimpleAvgForecast()
        }
        
        # Add models that need more data points
        if len(data) >= 3:
            models["Weighted Moving Average"] = WeightedMovingAvgForecast()
            models["Exponential Smoothing (α=0.3)"] = ExponentialSmoothingForecast(alpha=0.3)
            models["Exponential Smoothing (α=0.7)"] = ExponentialSmoothingForecast(alpha=0.7)
        
        if len(data) >= 4:
            models["Moving Average (Window=4)"] = MovingAvgForecast()
        
        if len(data) >= 5:
            models["Linear"] = LinearForecast()
        
        logger.debug(f"Initialized {len(models)} models")
        
        results = []
        for name, model in models.items():
            try:
                logger.debug(f"Running model: {name}")
                # Create a fresh copy of the data for each model
                model_data = [Observation(obs.period, obs.demand) for obs in data]
                
                # Run the model
                result = model.fit_predict(model_data, forecast_steps=forecast_steps)
                
                # Skip models that couldn't generate forecasts
                if not result or all(obs.forecast is None for obs in result):
                    logger.warning(f"Model {name} failed to generate forecasts")
                    continue
                
                # Calculate metrics
                metrics = calculate_metrics(data, result)
                mae = model.cal_mae(result)
                
                logger.debug(f"Model {name} metrics - MAE: {mae}, RMSE: {metrics['rmse']}, MAPE: {metrics['mape']}, R²: {metrics['r2']}")
                
                # Skip models with invalid metrics
                if mae is None or metrics['rmse'] is None or metrics['mape'] is None or metrics['r2'] is None:
                    logger.warning(f"Model {name} has invalid metrics")
                    continue
                
                # Skip models with infinite or NaN values
                if (np.isinf(mae) or np.isnan(mae) or 
                    np.isinf(metrics['rmse']) or np.isnan(metrics['rmse']) or 
                    np.isinf(metrics['mape']) or np.isnan(metrics['mape']) or 
                    np.isinf(metrics['r2']) or np.isnan(metrics['r2'])):
                    logger.warning(f"Model {name} has infinite or NaN metrics")
                    continue
                
                # Add metrics to result object
                for obs in result:
                    obs.metrics = metrics
                    obs.rmse = metrics['rmse']
                    obs.mape = metrics['mape']
                    obs.r2 = metrics['r2']
                
                results.append({
                    'name': name,
                    'forecasts': result,
                    'mae': mae,
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'r2': metrics['r2']
                })
                
                logger.debug(f"Model {name} completed successfully")
            except Exception as e:
                logger.error(f"Error in model {name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        if not results:
            logger.warning("No models were able to generate valid forecasts")
            return []
            
        # Sort results by MAE (lower is better)
        results.sort(key=lambda x: x['mae'])
        
        # Add status to each result
        for i, result in enumerate(results):
            if i == 0:
                result['status'] = 'Best Model'
            else:
                result['status'] = 'Alternative'
        
        logger.debug(f"Model comparison completed with {len(results)} valid results")
        return results
    except Exception as e:
        logger.error(f"Error in compare_models: {str(e)}")
        logger.error(traceback.format_exc())
        return []

@app.route("/export/<format>", methods=["POST"])
def export_results(format):
    """Export forecast results in the specified format"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Create DataFrame from the data
        df = pd.DataFrame(data)
        
        # Create a BytesIO object to store the file
        output = io.BytesIO()
        
        if format == 'csv':
            df.to_csv(output, index=False)
            mimetype = 'text/csv'
            filename = 'forecast_results.csv'
        elif format == 'excel':
            df.to_excel(output, index=False)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = 'forecast_results.xlsx'
        else:
            return jsonify({"error": "Unsupported format"}), 400
        
        output.seek(0)
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analysis', methods=['GET'])
def analysis():
    # Mock data for demonstration
    stats = {
        'mean': 150.5,
        'std': 25.3
    }
    seasonality = [7]  # Weekly seasonality
    outliers = [10, 25, 40]  # Example outlier indices
    model_results = [
        {
            'name': 'ARIMA',
            'mae': 12.5,
            'rmse': 15.3,
            'mape': 8.2,
            'r2': 0.85
        },
        {
            'name': 'Prophet',
            'mae': 13.2,
            'rmse': 16.1,
            'mape': 8.7,
            'r2': 0.82
        },
        {
            'name': 'LSTM',
            'mae': 11.8,
            'rmse': 14.9,
            'mape': 7.9,
            'r2': 0.87
        }
    ]
    
    # Mock charts
    charts = {
        'time_series': '<div id="timeSeriesChart"></div>',
        'distribution': '<div id="distributionChart"></div>'
    }
    
    return render_template('analysis.html',
                         stats=stats,
                         seasonality=seasonality,
                         outliers=outliers,
                         model_results=model_results,
                         charts=charts)

@app.route('/settings', methods=['GET'])
def settings():
    current_settings = load_settings()
    system_info = {
        'version': '1.0.0',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points': 1000,
        'models_available': 3,
        'storage_used': '2.5 MB'
    }
    return render_template('settings.html',
                         settings=current_settings,
                         **system_info)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    current_settings = load_settings()
    current_settings.update({
        'forecast_horizon': int(request.form.get('forecast_horizon', 30)),
        'confidence_interval': float(request.form.get('confidence_interval', 0.95)),
        'use_arima': 'use_arima' in request.form,
        'use_prophet': 'use_prophet' in request.form,
        'use_lstm': 'use_lstm' in request.form
    })
    save_settings(current_settings)
    return jsonify({'status': 'success'})

@app.route('/update_display_settings', methods=['POST'])
def update_display_settings():
    current_settings = load_settings()
    current_settings.update({
        'theme': request.form.get('theme', 'light'),
        'chart_style': request.form.get('chart_style', 'default'),
        'show_confidence': 'show_confidence' in request.form,
        'show_legend': 'show_legend' in request.form,
        'show_grid': 'show_grid' in request.form
    })
    save_settings(current_settings)
    return jsonify({'status': 'success'})

@app.route('/import_data', methods=['POST'])
def import_data():
    if 'data_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['data_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'status': 'success', 'message': 'File uploaded successfully'})

@app.route('/export_data', methods=['POST'])
def export_data():
    export_format = request.form.get('export_format', 'csv')
    # Mock data for demonstration
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'value': np.random.normal(100, 20, 100)
    })
    
    if export_format == 'csv':
        return send_file(
            data.to_csv(index=False),
            mimetype='text/csv',
            as_attachment=True,
            download_name='forecast_data.csv'
        )
    elif export_format == 'xlsx':
        return send_file(
            data.to_excel(index=False),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='forecast_data.xlsx'
        )
    elif export_format == 'json':
        return send_file(
            data.to_json(orient='records'),
            mimetype='application/json',
            as_attachment=True,
            download_name='forecast_data.json'
        )

if __name__ == "__main__":
    app.run(debug=True)
