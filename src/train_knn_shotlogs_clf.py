import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
    train, test = train_test_split(df, test_size=test_size, random_state=11)

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    train[:] = minmax_scaler.fit_transform(train[:].values)
    test[:] = minmax_scaler.transform(test[:].values)

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
        "n_neighbors": list(range(1, 21)),
    }

    model = KNeighborsClassifier(algorithm="auto")

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1, "./models/knnclf.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/knnclf_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)