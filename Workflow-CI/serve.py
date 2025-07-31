from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import joblib
import numpy as np

app = Flask(__name__)
metrics = PrometheusMetrics(app)

model = joblib.load("best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    X = np.array(data).reshape(1, -1)
    y_pred = model.predict(X)
    return jsonify({"prediction": int(y_pred[0])})

# /metrics sudah otomatis
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

