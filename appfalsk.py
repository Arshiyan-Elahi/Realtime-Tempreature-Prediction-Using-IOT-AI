from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from prophet import Prophet
import csv
import time
from datetime import datetime, timedelta
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
latest_data = {
    'temperature': None,
    'humidity': None,
    'last_update': None
}

DATA_TIMEOUT = 10  # Timeout in seconds

# Helper Functions
def save_to_csv(temperature, humidity):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open('realtime_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # If file is empty, write header
                writer.writerow(['timestamp', 'temperature', 'humidity'])
            writer.writerow([timestamp, temperature, humidity])
        print(f"Data saved to CSV: {timestamp}, {temperature}°C, {humidity}%")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def load_historical_data():
    try:
        df = pd.read_csv('realtime_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} historical records")
        return df
    except FileNotFoundError:
        print("No historical data file found. Creating new DataFrame.")
        return pd.DataFrame(columns=['timestamp', 'temperature', 'humidity'])
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame(columns=['timestamp', 'temperature', 'humidity'])

# Routes
@app.route('/')
def index():
    try:
        df = load_historical_data()
        initial_data = {
            'latest': latest_data,
            'has_data': not df.empty,
            'data_count': len(df)
        }
        return render_template('index.html', data=initial_data)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', error=str(e))

@app.route('/log_data', methods=['POST'])
def log_data():
    try:
        data = request.get_json()
        temperature = data.get('temperature')
        humidity = data.get('humidity')

        print(f"Received data: Temperature={temperature}°C, Humidity={humidity}%")

        latest_data.update({
            'temperature': temperature,
            'humidity': humidity,
            'last_update': time.time()
        })
        
        save_to_csv(temperature, humidity)
        
        return jsonify({
            "status": "success",
            "data": {
                "temperature": temperature,
                "humidity": humidity,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }), 200

    except Exception as e:
        print(f"Error in log_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/latest_data')
def get_latest_data():
    try:
        current_time = time.time()
        
        if latest_data['last_update'] is None:
            return jsonify({"error": "No data available"}), 404
        
        if (current_time - latest_data['last_update']) > DATA_TIMEOUT:
            return jsonify({"error": "Data is stale"}), 404
        
        print(f"Sending latest data: {latest_data}")
        return jsonify(latest_data)

    except Exception as e:
        print(f"Error in latest_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/historical_data')
def get_historical_data():
    try:
        df = load_historical_data()
        if df.empty:
            return jsonify({"error": "No historical data available"}), 404
            
        return jsonify(df.tail(100).to_dict('records'))

    except Exception as e:
        print(f"Error in historical_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/forecast')
def get_forecast():
    try:
        df = load_historical_data()
        if df.empty:
            return jsonify({"error": "Insufficient data for forecast"}), 404

        # Prepare data for Prophet
        prophet_df = df[['timestamp', 'temperature']].rename(
            columns={'timestamp': 'ds', 'temperature': 'y'}
        )
        
        # Configure and train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        
        model.fit(prophet_df)
        
        # Create forecast
        future = model.make_future_dataframe(periods=24, freq='h')
        forecast = model.predict(future)
        
        # Prepare forecast data
        forecast_data = {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24).to_dict('records'),
            'trend': forecast[['ds', 'trend']].to_dict('records'),
            'weekly': forecast[['ds', 'weekly']].to_dict('records'),
            'daily': forecast[['ds', 'daily']].to_dict('records')
        }
        
        print("Forecast generated successfully")
        return jsonify(forecast_data)

    except Exception as e:
        print(f"Error in forecast: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def get_stats():
    try:
        df = load_historical_data()
        
        if df.empty:
            return jsonify({"error": "No data available for stats"}), 404
            
        stats = {
            'current_temp': latest_data['temperature'],
            'current_humidity': latest_data['humidity'],
            'avg_temp': round(df['temperature'].mean(), 2),
            'max_temp': round(df['temperature'].max(), 2),
            'min_temp': round(df['temperature'].min(), 2),
            'avg_humidity': round(df['humidity'].mean(), 2),
            'records_count': len(df),
            'last_update': datetime.fromtimestamp(latest_data['last_update']).strftime('%Y-%m-%d %H:%M:%S') if latest_data['last_update'] else None
        }
        
        return jsonify(stats)

    except Exception as e:
        print(f"Error in stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/patterns')
def get_patterns():
    try:
        df = load_historical_data()
        
        if df.empty:
            return jsonify({"error": "No data available for pattern analysis"}), 404
        
        # Calculate patterns
        patterns = {
            'daily': df.groupby(df['timestamp'].dt.hour)['temperature'].mean().round(2).to_dict(),
            'weekly': df.groupby(df['timestamp'].dt.dayofweek)['temperature'].mean().round(2).to_dict(),
            'monthly': df.groupby(df['timestamp'].dt.month)['temperature'].mean().round(2).to_dict(),
            'humidity_daily': df.groupby(df['timestamp'].dt.hour)['humidity'].mean().round(2).to_dict(),
            'temperature_range': {
                'min': float(df['temperature'].min()),
                'max': float(df['temperature'].max()),
                'avg': float(df['temperature'].mean())
            }
        }
        
        return jsonify(patterns)

    except Exception as e:
        print(f"Error in patterns: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug')
def debug_info():
    """Route for debugging purposes"""
    try:
        df = load_historical_data()
        debug_data = {
            'latest_data': latest_data,
            'data_file_exists': True,
            'record_count': len(df),
            'last_records': df.tail(5).to_dict('records') if not df.empty else [],
            'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_update_age': time.time() - latest_data['last_update'] if latest_data['last_update'] else None
        }
        return jsonify(debug_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.43.86', port=5000, debug=True)