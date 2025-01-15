import os
import joblib
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

app.config['MODEL_FILE'] = 'model/uv_model_tf.h5'
app.config['SCALER_FILE'] = 'model/scaler.pkl'
app.config['LABELS_FILE'] = 'model/uv_index.txt'

try:
    model = load_model(app.config['MODEL_FILE'], compile=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

try:
    scaler = joblib.load(app.config['SCALER_FILE'])
except Exception as e:
    raise RuntimeError(f"Failed to load scaler: {e}")

try:
    with open(app.config['LABELS_FILE'], 'r') as file:
        labels = file.read().splitlines()
except Exception as e:
    raise RuntimeError(f"Failed to load labels: {e}")

def predict_uv_index(features):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_index = int(np.clip(prediction[0], 0, len(labels) - 1))
        return predicted_index
    except Exception as e:
        raise ValueError(f"Prediction error: {e}")

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Weather Prediction API is working.",
        },
        "data": {
            'Project_Name': 'UV Index Prediction',
            'Team': 'Cangcimen',
            'Anggota': [
                {'NPM': '065121068', 'Nama': 'Sandy Budi Wirawan', 'Universitas': 'Pakuan'},
                {'NPM': '065121083', 'Nama': 'Saidina Hikam', 'Universitas': 'Pakuan'},
                {'NPM': '065121076', 'Nama': 'M.Athar Kautsar', 'Universitas': 'Pakuan'},
                {'NPM': '065121077', 'Nama': 'M.Imam Fahrudin', 'Universitas': 'Pakuan'},
                {'NPM': '065121085', 'Nama': 'M.Leon Fadilah', 'Universitas': 'Pakuan'},
                {'NPM': '065121111', 'Nama': 'Eri Mustika Alam', 'Universitas': 'Pakuan'},
            ],
            'Created_By': 'Cangcimen Team',
            'CopyRight': '@2025 All Rights Reserved!'
        }
    }), 200

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "GET":
        return jsonify({
            "status": {
                "code": 200,
                "message": "Test Successful. Use POST method to make predictions."
            },
            "data": {
                "description": "To predict UV Index, use POST method with 'features' data."
            }
        }), 200

    elif request.method == "POST":
        try:
            data = request.get_json()

            if not data or 'features' not in data:
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "'features' not found in the request data"
                    },
                    "data": None
                }), 400

            features = data['features']

            if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
                return jsonify({
                    "status": {
                        "code": 400,
                        "message": "'features' must be a list of numerical values."
                    },
                    "data": None
                }), 400

            features_array = np.array(features).reshape(1, -1)

            predicted_index = predict_uv_index(features_array)

            uv_category = labels[predicted_index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success Predicting UV Index",
                },
                "data": {
                    "uv_index": predicted_index,
                    "uv_category": uv_category
                }
            }), 200

        except Exception as e:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": f"Error: {str(e)}",
                },
                "data": None,
            }), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
