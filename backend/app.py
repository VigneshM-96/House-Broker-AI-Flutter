from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# --------------------------
# Load model & encoder
# --------------------------
model = joblib.load("models/house_price_model.pkl")
location_map = joblib.load("models/location_encoder.pkl")

def encode_location(loc):
    return location_map.get(loc, -1)

# --------------------------
# Prediction endpoint
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = ["location", "area_sqft", "bedrooms", "bathrooms", "halls", "kitchens"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing fields"}), 400

        loc_encoded = encode_location(data["location"])
        if loc_encoded == -1:
            return jsonify({"error": f"Unknown location '{data['location']}'"}), 400

        features = np.array([[
            loc_encoded,
            int(data["area_sqft"]),
            int(data["bedrooms"]),
            int(data["bathrooms"]),
            int(data["halls"]),
            int(data["kitchens"])
        ]])

        prediction = model.predict(features)[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------
# Run app
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)