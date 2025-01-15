# UV Index Prediction REST API

## Overview

This API enables users to predict UV Index categories based on provided feature data. The prediction system utilizes machine learning models implemented with **TensorFlow** and **scikit-learn**. The API provides two main endpoints: **`/`** and **`/prediction`**.

## Key Features

* UV Index Prediction: Leverages machine learning to predict UV Index values from feature data
* Feature Scaling: Implements `StandardScaler` for normalizing input data before model prediction
* Category Labels: Returns UV Index categories based on predefined classification labels

## Technology Stack

* **Flask**: Python web framework for API development
* **TensorFlow**: Powers the machine learning model (`model/uv_model_tf.h5`)
* **scikit-learn**: Handles data scaling (`model/scaler.pkl`)
* **joblib**: Manages scaler model persistence
* **NumPy**: Enables numerical data manipulation

## Prerequisites

Before running the API, ensure you have:

* Python 3.x installed
* TensorFlow and scikit-learn packages installed
* Required model files in the `model/` directory:
  * `uv_model_tf.h5` (TensorFlow model)
  * `scaler.pkl` (Feature scaler)
  * `uv_index.txt` (Category labels)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify model files exist in the `model/` directory

## Running the API

Start the server with:

```bash
python app.py
```

The API will run at `http://0.0.0.0:8080`. You can configure the port through the `.env` file or command line.

## API Endpoints

### 1. GET /

Health check endpoint to verify API status.

**Request:**
```bash
GET http://localhost:8080/
```

**Response:**
```json
{
  "status": {
    "code": 200,
    "message": "Weather Prediction API is working."
  },
  "data": {
    "Project_Name": "UV Index Prediction",
    "Team": "Cangcimen",
    "Anggota": [
      {"NPM": "065121068", "Nama": "Sandy Budi Wirawan", "Universitas": "Pakuan"},
      {"NPM": "065121083", "Nama": "Saidina Hikam", "Universitas": "Pakuan"},
      {"NPM": "065121076", "Nama": "M.Athar Kautsar", "Universitas": "Pakuan"},
      {"NPM": "065121077", "Nama": "M.Imam Fahrudin", "Universitas": "Pakuan"},
      {"NPM": "065121085", "Nama": "M.Leon Fadilah", "Universitas": "Pakuan"},
      {"NPM": "065121111", "Nama": "Eri Mustika Alam", "Universitas": "Pakuan"}
    ],
    "Created_By": "Cangcimen Team",
    "CopyRight": "@2025 All Rights Reserved!"
  }
}
```

### 2. POST /prediction

Endpoint for UV Index predictions.

**Request:**
```bash
POST http://localhost:8080/prediction
Content-Type: application/json

{
  "features": [30.5, 45.2, 80, 100, 15.6]
}
```

**Success Response:**
```json
{
  "status": {
    "code": 200,
    "message": "Success Predicting UV Index"
  },
  "data": {
    "uv_index": 3,
    "uv_category": "High"
  }
}
```

**Error Response:**
```json
{
  "status": {
    "code": 400,
    "message": "'features' not found in the request data"
  },
  "data": null
}
```

## Error Handling

The API includes comprehensive error handling:

* **400 Bad Request**: Invalid or missing feature data
* **500 Internal Server Error**: Server-side processing errors

## Testing

Test the API using Postman or cURL:

1. Check API status:
   ```bash
   curl http://localhost:8080/
   ```

2. Make a prediction:
   ```bash
   curl -X POST http://localhost:8080/prediction \
        -H "Content-Type: application/json" \
        -d '{"features": [30.5, 45.2, 80, 100, 15.6]}'
   ```

## License

MIT License

---
Created by Cangcimen Team © 2025