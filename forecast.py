# forecast.py
import numpy as np
from collections import Counter
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class Observation:
    def __init__(self, period, demand):
        self.period = float(period) if period is not None else None
        self.demand = float(demand) if demand is not None else None
        self.forecast = None
        self.error = None
        self.metrics = None
        self.rmse = None
        self.mape = None
        self.r2 = None

    def __iter__(self):
        yield self

# Naive Forecast
class NaiveForecast:
    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < 2:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            # First period has no forecast
            data[0].forecast = None
            data[0].error = None
            
            # Calculate forecasts for historical data
            for i in range(1, len(data)):
                if data[i-1].demand is not None:
                    data[i].forecast = data[i-1].demand
                    if data[i].demand is not None:
                        data[i].error = data[i].demand - data[i].forecast
                    else:
                        data[i].error = None
                else:
                    data[i].forecast = None
                    data[i].error = None

            # Generate future forecasts
            last = data[-1]
            for i in range(forecast_steps):
                new_obs = Observation(last.period + 1, None)
                if last.demand is not None:
                    new_obs.forecast = last.demand
                else:
                    new_obs.forecast = None
                data.append(new_obs)
                last = new_obs
            return data
        except Exception as e:
            print(f"Error in NaiveForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

# Simple Average Forecast
class SimpleAvgForecast:
    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < 2:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            # First period has no forecast
            data[0].forecast = None
            data[0].error = None
            
            # Calculate forecasts for historical data
            for i in range(1, len(data)):
                past_demand = [obs.demand for obs in data[:i] if obs.demand is not None]
                if past_demand:
                    data[i].forecast = round(sum(past_demand) / len(past_demand), 2)
                    if data[i].demand is not None:
                        data[i].error = data[i].demand - data[i].forecast
                    else:
                        data[i].error = None
                else:
                    data[i].forecast = None
                    data[i].error = None

            # Generate future forecasts
            past_demand = [obs.demand for obs in data if obs.demand is not None]
            if past_demand:
                avg_forecast = round(sum(past_demand) / len(past_demand), 2)
                for step in range(forecast_steps):
                    next_period = data[-1].period + 1
                    new_obs = Observation(next_period, None)
                    new_obs.forecast = avg_forecast
                    data.append(new_obs)
            return data
        except Exception as e:
            print(f"Error in SimpleAvgForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

# Moving Average Forecast
class MovingAvgForecast:
    def __init__(self, window_size=4):
        self.window_size = window_size

    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < self.window_size:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            # First window_size periods have no forecast
            for i in range(self.window_size):
                data[i].forecast = None
                data[i].error = None
            
            # Calculate forecasts for historical data
            for i in range(self.window_size, len(data)):
                past_demand = [obs.demand for obs in data[i-self.window_size:i] if obs.demand is not None]
                if len(past_demand) == self.window_size:
                    data[i].forecast = round(sum(past_demand) / len(past_demand), 2)
                    if data[i].demand is not None:
                        data[i].error = data[i].demand - data[i].forecast
                    else:
                        data[i].error = None
                else:
                    data[i].forecast = None
                    data[i].error = None

            # Generate future forecasts
            for step in range(forecast_steps):
                next_period = data[-1].period + 1
                past_demand = [obs.demand for obs in data[-self.window_size:] if obs.demand is not None]
                if len(past_demand) == self.window_size:
                    next_forecast = round(sum(past_demand) / len(past_demand), 2)
                    new_obs = Observation(next_period, None)
                    new_obs.forecast = next_forecast
                    data.append(new_obs)
            return data
        except Exception as e:
            print(f"Error in MovingAvgForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

# Weighted Moving Average Forecast
class WeightedMovingAvgForecast:
    def __init__(self, weights=None):
        self.weights = weights if weights else [0.2, 0.3, 0.5]
        if len(self.weights) != 3:
            raise ValueError("Weights must be a list of 3 values")

    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < 3:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            # First 3 periods have no forecast
            for i in range(3):
                data[i].forecast = None
                data[i].error = None
            
            # Calculate forecasts for historical data
            for i in range(3, len(data)):
                past_demand = [obs.demand for obs in data[i-3:i] if obs.demand is not None]
                if len(past_demand) == 3:
                    data[i].forecast = round(sum(p * w for p, w in zip(past_demand, self.weights)), 2)
                    if data[i].demand is not None:
                        data[i].error = data[i].demand - data[i].forecast
                    else:
                        data[i].error = None
                else:
                    data[i].forecast = None
                    data[i].error = None

            # Generate future forecasts
            for step in range(forecast_steps):
                next_period = data[-1].period + 1
                past_demand = [obs.demand for obs in data[-3:] if obs.demand is not None]
                if len(past_demand) == 3:
                    next_forecast = round(sum(p * w for p, w in zip(past_demand, self.weights)), 2)
                    new_obs = Observation(next_period, None)
                    new_obs.forecast = next_forecast
                    data.append(new_obs)
            return data
        except Exception as e:
            print(f"Error in WeightedMovingAvgForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

# Exponential Smoothing Forecast
class ExponentialSmoothingForecast:
    def __init__(self, alpha=0.3):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha

    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < 2:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            # First period has no forecast
            data[0].forecast = None
            data[0].error = None
            
            # Initialize second period forecast
            if len(data) > 1 and data[0].demand is not None:
                data[1].forecast = data[0].demand
                if data[1].demand is not None:
                    data[1].error = data[1].demand - data[1].forecast
                else:
                    data[1].error = None
            else:
                data[1].forecast = None
                data[1].error = None
            
            # Calculate forecasts for historical data
            for i in range(2, len(data)):
                if data[i-1].demand is not None and data[i-1].forecast is not None:
                    data[i].forecast = round(self.alpha * data[i-1].demand + (1 - self.alpha) * data[i-1].forecast, 2)
                    if data[i].demand is not None:
                        data[i].error = data[i].demand - data[i].forecast
                    else:
                        data[i].error = None
                else:
                    data[i].forecast = None
                    data[i].error = None

            # Generate future forecasts
            last_forecast = data[-1].forecast
            last_demand = data[-1].demand
            for step in range(forecast_steps):
                next_period = data[-1].period + 1
                if last_demand is not None and last_forecast is not None:
                    next_forecast = round(self.alpha * last_demand + (1 - self.alpha) * last_forecast, 2)
                    new_obs = Observation(next_period, None)
                    new_obs.forecast = next_forecast
                    data.append(new_obs)
                    last_forecast = next_forecast
                    last_demand = next_forecast
            return data
        except Exception as e:
            print(f"Error in ExponentialSmoothingForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

# Linear Forecast
class LinearForecast:
    def fit_predict(self, data, forecast_steps=1):
        if not data or len(data) < 2:
            return []
            
        try:
            data = [Observation(obs.period, obs.demand) for obs in data]
            valid_data = [(obs.period, obs.demand) for obs in data if obs.demand is not None]
            
            if len(valid_data) < 2:
                return data
                
            x = np.array([period for period, _ in valid_data])
            y = np.array([demand for _, demand in valid_data])

            # Calculate linear regression parameters
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate forecasts for all periods
            for obs in data:
                if obs.period is not None:
                    obs.forecast = round(slope * obs.period + intercept, 2)
                    if obs.demand is not None:
                        obs.error = obs.demand - obs.forecast
                    else:
                        obs.error = None
                else:
                    obs.forecast = None
                    obs.error = None

            # Generate future forecasts
            last_period = data[-1].period
            for step in range(forecast_steps):
                next_period = last_period + 1
                new_obs = Observation(next_period, None)
                new_obs.forecast = round(slope * next_period + intercept, 2)
                data.append(new_obs)
                last_period = next_period
            return data
        except Exception as e:
            print(f"Error in LinearForecast: {str(e)}")
            return []

    def cal_mae(self, data):
        try:
            errors = [abs(obs.error) for obs in data if obs.error is not None]
            return round(sum(errors)/len(errors), 2) if errors else 0
        except Exception as e:
            print(f"Error calculating MAE: {str(e)}")
            return 0

class Forecast:
    def __init__(self, df, date_col, value_col):
        self.df = df
        self.date_col = date_col
        self.value_col = value_col
        self.model = None
        self.forecast_results = None
        
    def generate_forecast(self, periods=12, seasonality='additive', confidence_level=0.95, model_type='auto'):
        try:
            # Prepare data
            y = self.df[self.value_col].values
            
            # Select model based on type
            if model_type == 'auto':
                # Try different models and select the best one
                models = {
                    'simple': SimpleExpSmoothing(y).fit(),
                    'holt': Holt(y).fit(),
                    'holt-winters': ExponentialSmoothing(y, seasonal_periods=12, trend='add', seasonal=seasonality).fit()
                }
                
                # Calculate AIC for each model
                aic_scores = {name: model.aic for name, model in models.items()}
                best_model = min(aic_scores.items(), key=lambda x: x[1])[0]
                self.model = models[best_model]
            else:
                if model_type == 'simple':
                    self.model = SimpleExpSmoothing(y).fit()
                elif model_type == 'holt':
                    self.model = Holt(y).fit()
                elif model_type == 'holt-winters':
                    self.model = ExponentialSmoothing(y, seasonal_periods=12, trend='add', seasonal=seasonality).fit()
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate forecast
            forecast = self.model.forecast(periods)
            conf_int = self.model.get_prediction(start=len(y), end=len(y)+periods-1).conf_int(alpha=1-confidence_level)
            
            # Create forecast dataframe
            last_date = self.df[self.date_col].iloc[-1]
            forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq='M')[1:]
            
            self.forecast_results = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast,
                'lower_bound': conf_int[:, 0],
                'upper_bound': conf_int[:, 1]
            })
            
            return self.forecast_results
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")
    
    def calculate_rmse(self):
        if self.forecast_results is None:
            return None
        y_true = self.df[self.value_col].values[-len(self.forecast_results):]
        y_pred = self.forecast_results['forecast'].values
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self):
        if self.forecast_results is None:
            return None
        y_true = self.df[self.value_col].values[-len(self.forecast_results):]
        y_pred = self.forecast_results['forecast'].values
        return mean_absolute_error(y_true, y_pred)
    
    def calculate_mape(self):
        if self.forecast_results is None:
            return None
        y_true = self.df[self.value_col].values[-len(self.forecast_results):]
        y_pred = self.forecast_results['forecast'].values
        return mean_absolute_percentage_error(y_true, y_pred)