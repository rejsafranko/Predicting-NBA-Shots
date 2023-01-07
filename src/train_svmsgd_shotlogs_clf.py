import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from argparse import ArgumentParser
from joblib import dump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("test_size", type=float)
    return parser.parse_args()


def load_data(filename, test_size):
    df = pd.read_csv(filename, index_col=0)
    labels = df["SHOT_RESULT"]

    # Keep only numeric features.
    features = df.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    # Join features and labels.
    df = pd.concat([features, labels], axis=1)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size, random_state=2311)

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    train[:-1] = minmax_scaler.fit_transform(train[:-1].values)
    test[:-1] = minmax_scaler.transform(test[:-1].values)

    # Prepare dataset dictionaries.
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
    dataset = load_data(args.filename, args.test_size)

    # Optimal model selection.
    parameters_for_testing = {
        "loss": ["hinge", "log_loss", "squared_hinge", "modified_huber", "perceptron"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1", "elasticnet", "none"],
    }

    model = SGDClassifier(max_iter=2000, verbose=1)

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
        cv=2,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Print best parameters.
    print(gsearch1.best_params_)

    # Save the model.
    dump(gsearch1, "./models/sgdclassifier.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/sgdclassifier_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
