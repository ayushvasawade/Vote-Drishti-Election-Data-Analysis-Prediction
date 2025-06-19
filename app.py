from flask import Flask, request, jsonify, render_template

from verboselogs import VerboseLogger, VERBOSE
from coloredlogs import install as Cloginstall
from predict import predict_winner

logger = VerboseLogger('APP')
Cloginstall(level=VERBOSE, fmt='[%(asctime)s] | [%(name)s] | %(levelname)-8s | %(message)s')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid input format. Expected a list of candidate objects."}), 400

    prediction = predict_winner(data)
    return prediction

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
