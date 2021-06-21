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
    model_columns = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
       'grade', 'zipcode_98001', 'zipcode_98002', 'zipcode_98003',
       'zipcode_98004', 'zipcode_98005', 'zipcode_98006', 'zipcode_98007',
       'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014',
       'zipcode_98019', 'zipcode_98022', 'zipcode_98023', 'zipcode_98024',
       'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030',
       'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034',
       'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042',
       'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055',
       'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065',
       'zipcode_98070', 'zipcode_98072', 'zipcode_98074', 'zipcode_98075',
       'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103',
       'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108',
       'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116',
       'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122',
       'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136',
       'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155',
       'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178',
       'zipcode_98188', 'zipcode_98198', 'zipcode_98199', 'renovated',
       'dec_built_1900', 'dec_built_1910', 'dec_built_1920', 'dec_built_1930',
       'dec_built_1940', 'dec_built_1950', 'dec_built_1960', 'dec_built_1970',
       'dec_built_1980', 'dec_built_1990', 'dec_built_2000', 'dec_built_2010',
       'sqm_living', 'sqm_lot', 'sqm_above', 'sqm_basement']
    if model:
        try:
            json = request.get_json()
            values = np.array(json)
            x = pd.DataFrame(data=values)
            prediction = model.predict(x)
            return jsonify({'prediction': str(prediction[0])})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return 'No model here to use'


if __name__ == '__main__':
    app.run()
