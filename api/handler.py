import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.healthinsurance import HealthInsurance

#Loading Model
path = '/home/eron/repos/pa004_health_insurance_cross_sell/'
model = pickle.load(open(path + 'models/model_xgb_classifier.pkl', 'rb'))

#Initialize API
app = Flask (__name__)

@app.route('/predict', methods = ['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index = [0])
            
        else:
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        pipeline = HealthInsurance()
        
        data_clean = pipeline.data_cleaning(test_raw)
        
        data_eng = pipeline.data_engineering(data_clean)
        
        data_prep = pipeline.data_preparation(data_eng)
        
        data_response = pipeline.get_prediction(model, test_raw, data_prep)
        
        return data_response
    
    else:
        return Response('{}', status = 200, mimetype = 'application/json')
    
if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug = True)
