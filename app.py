from flask import Flask, jsonify, request
import pandas as pd
import joblib
import traceback

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    return "hello"


@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load("model.pkl")
    if model:
        try:
            json = request.get_json()
            x = pd.DataFrame(data=json, index=[0])
            prediction = model.predict(x)
            return jsonify({'prediction': str(prediction[0])})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return 'No model here to use'


if __name__ == '__main__':
    app.run()
