import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

# Load and prepare data
df = pd.read_csv('Delhi_v2.csv')

# Assume your data has these columns based on your previous details
X = df[['latitude', 'longitude', 'parking', 'Bedrooms', 'area', 'Bathrooms', 'Lift', 'Balcony', 'Furnished_status',
        'type_of_building', 'neworold']]
y = df['price']

# Convert categorical data to numerical (example)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
pickle.dump(model, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('proj.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])

    # Convert categorical input to numerical using get_dummies
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure input_data has the same columns as the training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0

    input_data = input_data[X.columns]  # Arrange columns in the same order

    prediction = model.predict(input_data)[0]

    return jsonify({'price': prediction})


if __name__ == '__main__':
    app.run(debug=True)
