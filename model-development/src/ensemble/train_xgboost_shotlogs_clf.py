from argparse import ArgumentParser

import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=False)
    return parser.parse_args()


def load_data(train_split: str, test_split: str):
    # Load csv files.
    train = pd.read_csv(train_split)
    test = pd.read_csv(test_split)

    # Prepare labels.
    y_train = train["SHOT_RESULT"]
    y_test = test["SHOT_RESULT"]

    # Drop label and unused features.
    X_train = train.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )
    X_test = test.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    # Prepare dataset dictionaries.
    dataset = dict()

    dataset["train"] = {
        "features": X_train,
        "labels": y_train,
    }
    dataset["test"] = {
        "features": X_test,
        "labels": y_test,
    }

    return dataset


def main(args):
    dataset = load_data(args.train_split, args.test_split)

    # Optimal model selection.
    parameters_for_testing = {
        "min_child_weight": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": [0.0001, 0.001],
        "n_estimators": [1, 3, 5, 10],
        "max_depth": [3, 4],
    }

    model = XGBClassifier()

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1.best_estimator_, "models/xgboost.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
