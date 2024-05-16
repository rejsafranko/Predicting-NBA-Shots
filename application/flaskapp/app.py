from flask import Flask, request, jsonify, Response
from sklearn.base import BaseEstimator
import joblib
import boto3
from io import BytesIO
from typing import Any, Dict

app = Flask(__name__)


def load_model_from_s3(bucket_name: str, object_name: str) -> BaseEstimator:
    """
    Load a machine learning model from an S3 bucket.

    :param bucket_name: Name of the S3 bucket.
    :param object_name: Key of the object within the S3 bucket.
    :return: Loaded model.
    """
    # Create a new S3 client using default credentials and configuration.
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=object_name)
    model_stream = BytesIO(response["Body"].read())
    model = joblib.load(model_stream)
    return model


BUCKET_NAME = "nba-models"
MODEL_FILE_NAME = "randomforest.joblib"
MODEL: BaseEstimator = load_model_from_s3(
    bucket_name=BUCKET_NAME, object_name=MODEL_FILE_NAME
)


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
