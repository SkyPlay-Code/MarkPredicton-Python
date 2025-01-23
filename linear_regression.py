import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample dataset creation
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'marks': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}
df = pd.DataFrame(data)

# Splitting the dataset into features and target variable
X = df[['study_hours']]
y = df['marks']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizing the results
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Student Marks Prediction')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()

# We can add attendance, previous grades, or participation in extracurricular activities to make the model more complex for better student marks prediction. We could also try using some other algorithms, for example, Decision Trees or Random Forests, which might work better than the given model. Here is an extended version of the code with an extended model using a Random Forest:

from sklearn.ensemble import RandomForestRegressor

# Sample dataset creation with additional features
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [80, 85, 90, 95, 70, 75, 80, 90, 95, 100],
    'previous_grades': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'marks': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}
df = pd.DataFrame(data)

# Splitting the dataset into features and target variable
X = df[['study_hours', 'attendance', 'previous_grades']]
y = df['marks']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizing the results
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with Random Forest')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

# To further enhance the model, we can implement feature scaling, hyperparameter tuning, and cross-validation. Feature scaling helps the model converge faster and improves accuracy. Here's how we can incorporate these enhancements:

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the scaled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Setting up the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Creating the model
model = RandomForestRegressor(random_state=42)

# Performing Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print(f'Best parameters: {grid_search.best_params_}')

# Making predictions with the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizing the results
plt.scatter(y_test, y_pred, color='purple')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with Optimized Random Forest')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

#You can also consider using additional evaluation metrics such as R-squared or Mean Absolute Error (MAE) to gain more insights into the model's performance.
from sklearn.metrics import r2_score, mean_absolute_error

# Evaluating the model with additional metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Visualizing the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='orange')
plt.axhline(0, linestyle='--', color='black')
plt.title('Residuals vs Predicted Marks')
plt.xlabel('Predicted Marks')
plt.ylabel('Residuals')
plt.show()

importances = grid_search.best_estimator_.feature_importances_
feature_names = X.columns

# Visualizing feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()

# By analyzing feature importance, you can make informed decisions about which features to keep or modify in your dataset, potentially leading to better model performance. You can also consider implementing techniques like cross-validation to ensure that your model generalizes well to unseen data. This can help in assessing the model's performance more reliably
from sklearn.model_selection import cross_val_score

# Performing cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = -cv_scores.mean()
print(f'Mean Cross-Validated MSE: {mean_cv_score}')

# Additionally, you might want to explore different regression algorithms beyond Random Forest, such as Support Vector Regression (SVR) or Gradient Boosting, which can sometimes yield better results depending on the dataset characteristics.
from sklearn.svm import SVR

# Creating the SVR model
svr_model = SVR(kernel='rbf')

# Training the SVR model
svr_model.fit(X_train, y_train)

# Making predictions
y_pred_svr = svr_model.predict(X_test)

# Evaluating the SVR model
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f'SVR Mean Squared Error: {mse_svr}')

# Visualizing the SVR results
plt.scatter(y_test, y_pred_svr, color='cyan')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with SVR')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

#Experimenting with different models and tuning their hyperparameters can lead to improved performance. We can also consider using ensemble methods that combine predictions from multiple models to enhance accuracy.
from sklearn.ensemble import VotingRegressor

# Creating individual models
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model2 = SVR(kernel='rbf')

# Creating the voting regressor
voting_model = VotingRegressor(estimators=[('rf', model1), ('svr', model2)])

# Training the voting regressor
voting_model.fit(X_train, y_train)

# Making predictions
y_pred_voting = voting_model.predict(X_test)

# Evaluating the voting regressor
mse_voting = mean_squared_error(y_test, y_pred_voting)
print(f'Voting Regressor Mean Squared Error: {mse_voting}')

# Visualizing the voting regressor results
plt.scatter(y_test, y_pred_voting, color='magenta')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with Voting Regressor')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

# By exploring various models and techniques, we can find the best approach for our specific dataset and prediction task. We can also consider implementing a pipeline to streamline the process of data preprocessing, model training, and evaluation. This can help in maintaining clean and organized code.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Defining the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['study_hours', 'attendance', 'previous_grades'])
    ])

# Creating a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training the pipeline
pipeline.fit(X_train, y_train)

# Making predictions
y_pred_pipeline = pipeline.predict(X_test)

# Evaluating the pipeline model
mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
print(f'Pipeline Mean Squared Error: {mse_pipeline}')

# Visualizing the pipeline results
plt.scatter(y_test, y_pred_pipeline, color='teal')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with Pipeline')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

# Using a pipeline not only simplifies the code but also ensures that the same preprocessing steps are applied consistently during both training and testing. Additionally, we can save the trained model using joblib or pickle for future use without needing to retrain it. 
import joblib

# Saving the trained pipeline model
joblib.dump(pipeline, 'student_marks_prediction_model.pkl')

# Loading the model
loaded_model = joblib.load('student_marks_prediction_model.pkl')

# Making predictions with the loaded model
y_pred_loaded = loaded_model.predict(X_test)

# Evaluating the loaded model
mse_loaded = mean_squared_error(y_test, y_pred_loaded)
print(f'Loaded Model Mean Squared Error: {mse_loaded}')

# We can also consider implementing a web application using Flask or Streamlit to create an interactive interface for users to input their data and receive predictions.
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array([[data['study_hours'], data['attendance'], data['previous_grades']]])
    prediction = loaded_model.predict(input_data)
    return jsonify({'predicted_marks': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

#This Flask app listens for POST requests at the /predict endpoint, takes input data in JSON format, and returns the predicted marks. WE can test this API using tools like Postman or cURL. By integrating our model into a web application, We can make it accessible to a wider audience and provide a user-friendly experience. We can also enhance the web application by adding input validation and error handling to ensure that the data received is in the correct format. This can help prevent issues when users submit invalid data.
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Input validation
    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_grades = float(data['previous_grades'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input data. Please provide study_hours, attendance, and previous_grades as numbers.'}), 400

    input_data = np.array([[study_hours, attendance, previous_grades]])
    prediction = loaded_model.predict(input_data)
    return jsonify({'predicted_marks': prediction[0]})

from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    study_hours = db.Column(db.Float, nullable=False)
    attendance = db.Column(db.Float, nullable=False)
    previous_grades = db.Column(db.Float, nullable=False)
    predicted_marks = db.Column(db.Float, nullable=False)

# Create the database tables
db.create_all()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Input validation
    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_grades = float(data['previous_grades'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input data. Please provide study_hours, attendance, and previous_grades as numbers.'}), 400
    
    input_data = np.array([[study_hours, attendance, previous_grades]])
    prediction = loaded_model.predict(input_data)

    # Save to database
    # Assuming a user with ID 1 for demonstration purposes
    new_prediction = Prediction(user_id=1, study_hours=study_hours,
                                attendance=attendance, previous_grades=previous_grades,
                                predicted_marks=prediction[0])
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({'predicted_marks': prediction[0]})

import pytest

def test_predict():
    client = app.test_client()
    response = client.post('/predict', json={
        'study_hours': 5,
        'attendance': 90,
        'previous_grades': 75
    })
    assert response.status_code == 200
    assert 'predicted_marks' in response.get_json()