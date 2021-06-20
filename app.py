from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback

app = Flask(__name__)


@app.route("/", methods=['GET'])
def hello():
    return "hello"


@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    if model:
        try:
            json = request.get_json()
            return json
            values = list(json[0].values())
            values = np.array(values)
            x = pd.DataFrame(data=values, columns=model_columns)
            prediction = model.predict(x)
            print("here:", prediction)
            return jsonify({'prediction': str(prediction[0])})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return 'No model here to use'


if __name__ == '__main__':
    app.run()
