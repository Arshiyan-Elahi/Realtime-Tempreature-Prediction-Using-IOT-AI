import dht
from machine import Pin
import time
import urequests
import network

# Initialize DHT11 sensor on GPIO pin 25
sensor = dht.DHT11(Pin(25))

# Wi-Fi connection details
ssid = 'NO WIFI SIGNAL 2G'  # Replace with your Wi-Fi SSID
password = 'p@ssw0rd'       # Replace with your Wi-Fi password

# Connect to Wi-Fi
station = network.WLAN(network.STA_IF)
station.active(True)
station.connect(ssid, password)

# Wait until connected
while not station.isconnected():
    print("Connecting to Wi-Fi...")
    time.sleep(1)
print("Connected to Wi-Fi:", station.ifconfig())

# URL for your Flask server (replace with correct IP)
url = "http://192.168.43.86:5000/log_data"

# Main loop to send data every 5 minutes or if temperature >= 30°C
while True:
    try:
        # Read sensor data
        sensor.measure()
        temp = sensor.temperature()
        humidity = sensor.humidity()

        # Check if temperature is 30°C or above
        if temp >= 25:
            # Send data to the Flask server
            response = urequests.post(url, json={'temperature': temp, 'humidity': humidity})
            if response.status_code == 200:
                print("Data sent successfully!")
            else:
                print(f"Error sending data: {response.status_code}")
            response.close()
        else:
            print(f"Temperature is {temp}C.")  # Use simple characters without extended symbols
            

    except OSError as e:
        print("Sensor or Network error:", e)

    # wait for 5 minutes = 300 sec
    time.sleep(10)
