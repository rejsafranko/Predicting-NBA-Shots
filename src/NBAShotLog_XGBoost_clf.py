import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
import math
from argparse import ArgumentParser
from joblib import dump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("test_size", type=float)
    return parser.parse_args()


def load_dataset(filename, test_size):
    df = pd.read_csv(filename, index_col=0)
    labels = df["SHOT_RESULT"]

    # Keep only numeric features.
    features = df.drop(
        [
            "LOCATION",
            "SHOT_NUMBER",
            "GAME_CLOCK",
            "CLOSEST_DEFENDER_PLAYER_ID",
            "player_id",
            "SHOT_RESULT",
        ],
        axis=1,
    )

    # Join features and labels.
    df = pd.concat([features, labels], axis=1)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size)
    dataset["train"] = {
        "features": train.loc[:, train.columns != "SHOT_RESULT"],
        "labels": train["SHOT_RESULT"],
    }
    dataset["test"] = {
        "features": test.loc[:, test.columns != "SHOT_RESULT"],
        "labels": test["SHOT_RESULT"],
    }

    return dataset


def main(args):
    dataset = load_dataset(args.filename, args.test_size)

    # Train initial model.
    xgb_model = XGBClassifier().fit(
        dataset["train"]["features"], dataset["train"]["labels"], verbose=1
    )

    # Optimal parameter search.
    parameters_for_testing = {
        "min_child_weight": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": [0.0001, 0.001],
        "n_estimators": [1, 3, 5, 10],
        "max_depth": [3, 4],
    }

    xgb_model = XGBClassifier()

    gsearch1 = GridSearchCV(
        estimator=xgb_model,
        param_grid=parameters_for_testing,
        scoring="precision",
        verbose=1,
    )

    # Train fine-tuned model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1, "./models/xgboost.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/xgboost_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
