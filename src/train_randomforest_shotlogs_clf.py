import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

    # Remove categoric data and labels.
    features = df.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    # Join features and labels.
    df = pd.concat([features, labels], axis=1)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size, random_state=11)

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
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }

    model = RandomForestClassifier(verbose=1)

    gsearch1 = GridSearchCV(
        estimator=model,
        param_grid=parameters_for_testing,
        scoring="accuracy",
        verbose=1,
        cv=2,
    )

    # Train the model.
    gsearch1.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the model.
    dump(gsearch1, "./models/randomforest.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/randomforest_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
