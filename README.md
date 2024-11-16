# Realtime-Tempreature-Prediction-Using-IOT-AI

This project demonstrates a real-time temperature prediction system integrating Flask, Streamlit, and ESP32. The system is designed to collect temperature data from an ESP32 sensor, process it using a machine learning model, and display predictions on an interactive web interface.

Key Features:
ESP32 Integration: Captures live temperature data and transmits it to the server.
Flask Backend: Handles data processing and prediction logic, ensuring smooth communication between the ESP32 and web interface.
Streamlit Dashboard: Provides an intuitive and interactive visualization of real-time temperature data and predictions.
Machine Learning Model: Predicts future temperature trends based on live sensor data.
Scalable Design: The system can be easily extended to incorporate additional sensors or prediction models.

How It Works:
ESP32 collects temperature readings and sends them to the Flask backend.
Flask processes the data and runs it through a trained machine learning model.
Predictions are forwarded to the Streamlit app, which displays the data and trends in real-time.

Technologies Used:
Hardware: ESP32 with temperature sensor.
Backend: Python, Flask.
Frontend: Streamlit for web visualization.
Machine Learning:Prophet model for temperature prediction.
