# Testing the Flask app
# Note: This should be in a separate test file, not in the main application code.
def test_predict():
    client = app.test_client()
    response = client.post('/predict', json={
        'study_hours': 5,
        'attendance': 90,
        'previous_grades': 75
    })
    assert response.status_code == 200
    assert 'predicted_marks' in response.get_json()