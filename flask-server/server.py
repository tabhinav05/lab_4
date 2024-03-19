from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_weight_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Extract features from input data
        length1 = float(data['length1'])
        length2 = float(data['length2'])
        length3 = float(data['length3'])
        height = float(data['height'])
        width = float(data['width'])
        
        # Make prediction
        prediction = model.predict([[length1, length2, length3, height, width]])
        
        # Prepare response
        response = {'prediction': prediction[0]}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)