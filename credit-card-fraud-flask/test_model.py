import joblib
import numpy as np

m = joblib.load('model/fraud_model.pkl')
print('model type:', type(m))
print('predict:', m.predict(np.zeros((1,30))))
if hasattr(m, 'predict_proba'):
    print('proba:', m.predict_proba(np.zeros((1,30)))[0])
else:
    print('no predict_proba')
