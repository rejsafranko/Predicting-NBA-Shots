import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

    # Normalize features.
    features[:] = MinMaxScaler().fit_transform(features.values)

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
    dataset = load_data(args.filename, args.test_size)
    model = SVC(verbose=1)

    # Training the model.
    model.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Measure prediction accuracy.
    score = model.score(dataset["test"]["features"], dataset["test"]["labels"])
    print(score)

    # Save the model.
    dump(model, "./models/svm.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/svm_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
