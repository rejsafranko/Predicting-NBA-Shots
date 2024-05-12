import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
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

    # Remove labels and excess features.
    features = df.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    # Join features and labels.
    df = pd.concat([features, labels], axis=1)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size, random_state=11)
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

    # Optimal model selection.
    parameters_for_testing = {
        "min_child_weight": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": [0.0001, 0.001],
        "n_estimators": [1, 3, 5, 10],
        "max_depth": [5, 10],
    }

    model = LGBMClassifier()

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
        cv=5,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1, "./models/lightgbm.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/lightgbm_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
