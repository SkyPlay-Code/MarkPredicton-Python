import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import matplotlib.pyplot as plt

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

# Creating a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training the pipeline
pipeline.fit(X_train, y_train)

# Making predictions
y_pred = pipeline.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Visualizing the results
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Student Marks Prediction with Random Forest')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.show()

# Saving the trained pipeline model
joblib.dump(pipeline, 'student_marks_prediction_model.pkl')

# Flask app setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_hours = db.Column(db.Float, nullable=False)
    attendance = db.Column(db.Float, nullable=False)
    previous_grades = db.Column(db.Float, nullable=False)
    predicted_marks = db.Column(db.Float, nullable=False)

# Create the database tables
with app.app_context():
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
    prediction = pipeline.predict(input_data)

    # Save to database
    new_prediction = Prediction(study_hours=study_hours,
                                attendance=attendance,
                                previous_grades=previous_grades,
                                predicted_marks=prediction[0])
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({'predicted_marks': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)