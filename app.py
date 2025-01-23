from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Input validation
    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_grades = float(data['previous_grades'])
        
        # Mock prediction logic
        # For example, let's say predicted marks = (study_hours * 10) + (attendance * 0.5) + (previous_grades * 0.5)
        predicted_marks = (study_hours * 10) + (attendance * 0.5) + (previous_grades * 0.5)
        
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input data. Please provide valid numbers.'}), 400

    return jsonify({'predicted_marks': predicted_marks})

if __name__ == '__main__':
    app.run(debug=True)