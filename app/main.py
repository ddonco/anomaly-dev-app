import glob
import numpy as np
import pickle
import sys
import xgboost
from flask import Flask, json, request, jsonify
from sklearn.cluster import DBSCAN
from tensorflow import keras

app = Flask(__name__)

nn_model_files = glob.glob('nn_model/*')
print(nn_model_files)
nn_model = keras.models.load_model(nn_model_files[0])
nn_std = 12
nn_tuning_param = 1
nn_tuning_factor = 1 / nn_tuning_param

xgb_model_files = glob.glob('xgb_model/*')
print(xgb_model_files)
xgb_model = pickle.load(open(xgb_model_files[0], 'rb'))
xgb_std = 10
xgb_tuning_param = 1
xgb_tuning_factor = 1 / nn_tuning_param

ss_airspeed_sp_threshold = 2

@app.route('/')
def home():
    return {}

@app.route('/nn-predict', methods=['POST'])
def nn_predict():
    req = request.get_json()
    
    print(f'nn model predict request: {req}', file=sys.stdout)
    x = np.expand_dims(np.array(req['x']), 0)
    y = req['y']
    y_pred = nn_model.predict(x)
    upper_confidence_interval = y_pred[0] + nn_std * nn_tuning_factor
    lower_configence_interval = y_pred[0] - nn_std * nn_tuning_factor

    anomaly = 0
    if y > upper_confidence_interval or y < lower_configence_interval:
        anomaly = 1

    return jsonify({'anomaly': [anomaly], 'y': y, 'y_pred': np.squeeze(y_pred, 1).tolist()})

@app.route('/xgb-predict', methods=['POST'])
def xgb_predict():
    req = request.get_json()
    
    print(f'xgb model predict request: {req}', file=sys.stdout)
    x = np.expand_dims(np.array(req['x']), 0)
    y = req['y']
    y_pred = xgb_model.predict(x)
    upper_confidence_interval = y_pred[0] + xgb_std * xgb_tuning_factor
    lower_configence_interval = y_pred[0] - xgb_std * xgb_tuning_factor

    anomaly = 0
    if y > upper_confidence_interval or y < lower_configence_interval:
        anomaly = 1

    return jsonify({'anomaly': [anomaly], 'y': y, 'y_pred': y_pred.tolist()})

@app.route('/dbscan-ss-classify', methods=['POST'])
def dbscan_ss_classify():
    dbscan_model = DBSCAN(eps=30, min_samples=5)
    req = request.get_json()
    
    print(f'dbscan model classify request: {req}', file=sys.stdout)
    x = np.array(req['x'])
    
    if len(x.shape) == 1:
        x = np.array(req['x']).reshape(-1, 1)
    elif len(x.shape) > 2:
        return jsonify({'error': f'x has incorrect shape ({x.shape}) for dbscan classification, shape must be 1 or 2 dimensional array'})
        
    dbscan_model.fit(x)
    labels = dbscan_model.labels_
    anomaly = np.where(labels < 0, 1, 0)

    return jsonify({'anomaly': anomaly.tolist(), 'x': req['x']})

@app.route('/dbscan-classify', methods=['POST'])
def dbscan_classify():
    dbscan_model = DBSCAN(eps=30, min_samples=5)
    req = request.get_json()
    
    print(f'dbscan model classify request: {req}', file=sys.stdout)
    x = req['x']
    timestamp = np.array(x['timestamp'])
    airspeed = np.array(x['Airspeed'])
    airspeed_sp = np.array(x['AirspeedSetpoint'])
    power = np.array(x['Power'])
    air_temperature = np.array(x['AirTemperature'])
    
    airspeed[airspeed == 0.] = 0.0001
    airspeed_pid_error = np.absolute((airspeed - airspeed_sp) / airspeed) * 100
    airspeed_stable = np.where(airspeed_pid_error < 2, 1, 0)

    system_ss = np.full(airspeed.shape, False)
    for i in range(10, len(airspeed_stable)):
        if (abs(airspeed_sp[i] - airspeed_sp[i - 1]) < ss_airspeed_sp_threshold and
            (np.all((airspeed_stable[i - 10:i] == 1)) or system_ss[i - 1] == True)):
            
            system_ss[i] = True
    
    system_ss = np.array(system_ss)
    ss_timestamp = timestamp.reshape(-1, 1)[system_ss, :]
    ss_airspeed = airspeed.reshape(-1, 1)[system_ss, :]
    ss_power = power.reshape(-1, 1)[system_ss, :]
    ss_airtemperature = air_temperature.reshape(-1, 1)[system_ss, :]
    ss_data = np.concatenate((ss_power, ss_airspeed, ss_airtemperature), axis=1)
    
    if len(ss_data.shape) > 2:
        return jsonify({'error': f'Data array has incorrect shape ({ss_data.shape}) for dbscan classification, shape must be 1 or 2 dimensional array'})
        
    dbscan_model.fit(ss_data)
    labels = dbscan_model.labels_
    anomaly = np.where(labels < 0, 1, 0)

    return jsonify(
        {
            'anomaly': anomaly.tolist(), 
            'steadystate': system_ss.tolist(), 
            'steadystate_timestamps': np.squeeze(ss_timestamp).tolist(),
            'steadystate_data': ss_data.tolist()
        }
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)