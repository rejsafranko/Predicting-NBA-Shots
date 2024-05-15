from argparse import ArgumentParser

import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


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

    # Keep only numeric features.
    X_train = train.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )
    X_test = test.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    X_train[:] = minmax_scaler.fit_transform(X_train[:].values)
    X_test[:] = minmax_scaler.transform(X_test[:].values)

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
        "alpha": [0.001, 0.01, 0.1, 1, 10],
    }

    model = MultinomialNB()

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1.best_estimator_, "models/multinomialnb.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
