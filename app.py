from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from flask import Flask, render_template
import os

from flask import Flask, render_template

app = Flask(__name__)
@app.route('/')
def home():
    # ML code here
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Load the training dataset
    data = pd.read_csv("weather_dataset.csv")

    # Split the data into input features (X) and target variable (y)
    X = data[['Temperature', 'Humidity', 'Pressure']]
    y = data['Weather']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Support Vector Machine (SVM) classifier
    model = SVC()   
    model.fit(X_train, y_train)

    # Load the data for prediction
    predict_data = pd.read_csv("feed.csv")
    predict_X = predict_data[['Temperature', 'Humidity', 'Pressure']]

    #Column for data and time
    date_time = predict_data.iloc[:, 0]

    # Standardize the input features for prediction
    predict_X = scaler.transform(predict_X)

    # Make predictions on the unseen data
    predictions = model.predict(predict_X)

    # Print the predicted weather
    print("Predicted weather:")
    for prediction in predictions:
        print(prediction)

    # Return the predicted weather as a list
    predicted_weather = {}
    for i in range(len(predictions)):
        predicted_weather[date_time[i]] = predictions[i]

    return render_template('index.html', predicted_weather=predicted_weather)


if __name__ == '__main__':
    app.run(debug=True)

