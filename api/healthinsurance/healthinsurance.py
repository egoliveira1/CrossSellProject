import pickle
import pandas as pd
import numpy as np
import inflection

class HealthInsurance(object):
    
    def __init__ (self):
        self.home_path             = '/home/eron/repos/pa004_health_insurance_cross_sell/'
        self.age_scaler            = pickle.load(open(self.home_path + 'src/features/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler = pickle.load(open(self.home_path + 'src/features/annual_premium_scaler.pkl', 'rb'))
        self.gender_scaler         = pickle.load(open(self.home_path + 'src/features/gender_scaler.pkl', 'rb'))
        self.policy_sales_scaler   = pickle.load(open(self.home_path + 'src/features/policy_sales_scaler.pkl', 'rb'))
        self.region_code_scaler    = pickle.load(open(self.home_path + 'src/features/region_code_scaler.pkl', 'rb'))
        self.vintage_scaler        = pickle.load(open(self.home_path + 'src/features/vintage_scaler.pkl', 'rb'))
        self.vehicle_age_scaler    = pickle.load(open(self.home_path + 'src/features/vehicle_age_scaler.pkl', 'rb'))
        
    
    def data_cleaning (self, data_clean):
        # Change columns name
        cols_old = data_clean.columns
        snakecase = lambda x: inflection.underscore (x)
        cols_new = list(map(snakecase, cols_old))
        data_clean.columns = cols_new
        
        # Cheange columns type        
        data_clean['region_code'] = data_clean['region_code'].astype(np.int64)
        data_clean['policy_sales_channel'] = data_clean['policy_sales_channel'].astype(np.int64)
        
        return data_clean
    
    
    def data_engineering (self, data_eng):
        data_eng['vehicle_age'] = data_eng['vehicle_age'].apply( lambda x: 3 if x == '> 2 Years' else 2 if x == '1-2 Year' else 1)
        data_eng['vehicle_damage'] = data_eng['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        data_eng['gender'] = data_eng['gender'].apply( lambda x: 1 if x == 'Female' else 0)
        damage_per_rcode = data_eng[['vehicle_damage']].groupby(data_eng['region_code']).mean().reset_index()
        damage_per_rcode = damage_per_rcode.rename( columns = {'vehicle_damage' : 'damage_per_rcode'})
        damage_per_rcode['damage_per_rcode'] = damage_per_rcode['damage_per_rcode'].astype(np.float64)
        data_eng = data_eng.merge(damage_per_rcode, on = 'region_code', how = 'left')
        
        return data_eng
        
        
    def data_preparation (self, data_prep):
        data_prep.loc[:, 'gender'] = data_prep.loc[:, 'gender'].map(self.gender_scaler)
        data_prep.loc[:, 'age'] = self.age_scaler.transform(data_prep[['age']].values)
        data_prep.loc[:, 'region_code'] = data_prep.loc[:, 'region_code'].map(self.region_code_scaler)
        data_prep.loc[:, 'vehicle_age'] = data_prep.loc[:, 'vehicle_age'].map(self.vehicle_age_scaler)
        data_prep.loc[:, 'annual_premium'] = self.annual_premium_scaler.transform(data_prep[['annual_premium']].values)
        data_prep.loc[:, 'policy_sales_channel'] = data_prep['policy_sales_channel'].map(self.policy_sales_scaler)
        data_prep.loc[:, 'vintage'] = self.vintage_scaler.transform(data_prep[['vintage']].values)
        cols_selected = ['vehicle_damage', 'policy_sales_channel', 'vehicle_age', 'previously_insured', 
                         'vintage', 'annual_premium', 'age', 'region_code', 'damage_per_rcode'] 
        
        return data_prep[cols_selected]
    
    
    def get_prediction (self, model, original_data, test_data):
        
        pred = model.predict_proba(test_data)
        original_data['score'] = pred[:,1].tolist()
        
        return original_data.to_json(orient = 'records', date_format = 'iso')
