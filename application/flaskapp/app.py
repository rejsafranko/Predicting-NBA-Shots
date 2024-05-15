from flask import Flask, request, jsonify, Response
from sklearn.base import BaseEstimator
import joblib

from typing import Any, Dict

app = Flask(__name__)

MODEL: BaseEstimator = joblib.load("path_to_model")


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        features: list = data["features"]

        if not isinstance(features, list):
            raise ValueError("Features must be a list.")

        # Model expect a single array of features.
        prediction = MODEL.predict([data["features"]])
        return jsonify({"prediction": list(prediction)})

    except KeyError:
        return jsonify({"error": "Data must include 'features'."}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Generic error handler for any other exceptions.
        return jsonify({"error": "An error occurred during prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True)
