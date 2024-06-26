# Weather-Prediction-App

The Python Flask framework was used to create the web-based Weather Prediction App. Users can predict the weather with this app. This program uses the Random Forest algorithm, a machine learning technique, to predict weather patterns. Based on a variety of meteorological characteristics and known weather data, it offers customers reliable predictions for the weather.

# Machine Learning and Data Variables

Several important meteorological variables, such as  "Temperature," "Dew," "Humidity," "Wind Speed(in km) ," "Visibility(in km)," and "Pressure," were used in the training of the machine learning model. Predicting the target variable "Weather" using these features. The 'Weather' variable originally included more than 50 categories. But the number of categories were reduced down to 9 in order to enhance the accuracy and performance of the model. Due to the lack of data points in many categories, which may limit the model's accuracy, the categories containing less than 100 data points were removed which gave us the total of 9 target variables. 

# Data Cleaning and Preparation

To make sure the dataset was appropriate for training the Random Forest model, a thorough cleaning process was carried out. Managing missing values, normalizing the data, and making sure that each category of the target variable was fairly represented. After then, the data was divided into training and testing sets making sure that all the target categories get split with the given ratio. This  was essential to create a model that would perform well when applied to new data.

# Model Training and Accuracy

A reliable and precise prediction was made using the Random Forest technique; the model's accuracy was 72%. This accuracy rating indicates an average forecast skill, given the complexity and variety of meteorological data.

# Model Deployment
To make the model accessible and usable for end-users, the trained Random Forest model was serialized using Python's pickle module. This allowed the model to be saved and loaded efficiently, facilitating its integration into the web application. The frontend of the app was built using HTML, providing a user-friendly interface for interacting with the weather prediction system.

# Web Application Development

The web application was developed using Flask. Flask made it possible for the machine learning model to be seamlessly integrated with the web interface, giving users the ability to input data and get weather predictions. 

# Conclusion 

The Weather Prediction App is an example of how machine learning is applied effectively in real-world scenarios. With the help of the Random Forest algorithm and a focus on important meteorological factors, the app provides precise weather predictions.  
