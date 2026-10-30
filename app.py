
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)#call constructor ans assign the app

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Iris Species Prediction API!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    s_l = data["sepal_length"]
    s_w = data["sepal_width"]
    p_l = data["petal_length"]
    p_w = data["petal_width"]

    pred = model.predict([[s_l, s_w, p_l, p_w]])

    return jsonify({"prediction": str(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
