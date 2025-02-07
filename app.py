from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('sales_predictor_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        input_data = request.get_json()

        input_features = np.array([[
            input_data['Quantity'],
            input_data['Price'],
            input_data['Customer_Age'],
            input_data['Family_Size'],
            input_data['Month']
        ]])

        prediction = model.predict(input_features)

        return jsonify({'predicted_sales': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
