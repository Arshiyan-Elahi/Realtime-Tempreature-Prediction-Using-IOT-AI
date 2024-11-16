# Real-Time Temperature Prediction System
This project demonstrates a real-time temperature prediction system using Flask, Streamlit, and ESP32. The system collects temperature data from an ESP32 sensor, processes it with a machine learning model, and visualizes predictions on an interactive web interface.

## Key Features
ESP32 Integration:
- Captures live temperature data from a connected sensor.
- Sends data to the backend server for processing.
Flask Backend:
- Handles data communication and processing.
- Runs the trained machine learning model to predict temperature trends.
Streamlit Dashboard:
- Displays real-time data in an interactive and user-friendly interface.
- Visualizes predictions and historical trends for better analysis.
Machine Learning Model:
- Uses the Prophet model for temperature prediction.
- Provides accurate short-term forecasts based on live sensor data.
Scalability:
- Easily extendable to include multiple sensors or additional environmental metrics.

## How It Works
- The ESP32 captures live temperature readings and sends them to the Flask server via an HTTP request.
- The Flask backend processes the incoming data and runs it through the trained Prophet model.
- The processed data and predictions are sent to the Streamlit dashboard.
- The Streamlit interface displays real-time temperature readings, prediction trends, and historical data visualization.

## Technologies Used  
- Hardware: ESP32 with a temperature sensor (e.g., DHT11 or DS18B20).
- Backend: Flask (Python) for handling requests and prediction logic.
- Frontend: Streamlit for interactive and dynamic visualization.
- Machine Learning: Prophet model for predictive analytics.
