from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained Random Forest model
model = pickle.load(open('Flask_RF.pkl', 'rb'))  # Update with your model file path

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a POST method for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    temperature = float(request.form['temperature'])
    dew = float(request.form['dew'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    visibility_km = float(request.form['visibility_km'])
    pressure = float(request.form['pressure'])  # New input field

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Dew': [dew],
        'Humidity': [humidity],
        'Wind_Speed': [wind_speed],
        'Visibility_km': [visibility_km],
        'Pressure': [pressure]  # Include Pressure in the DataFrame
    })

    # Make prediction using the model
    prediction = model.predict(input_data)

    # Return the predicted value
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)