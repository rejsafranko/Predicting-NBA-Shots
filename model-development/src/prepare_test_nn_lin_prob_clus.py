from argparse import ArgumentParser

import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)
    return parser.parse_args()


def main(args):
    # Load csv files.
    train = pd.read_csv(args.train_split)
    test = pd.read_csv(args.test_split)

    # Prepare labels.
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
    minmax_scaler.fit(X_train[:].values)
    X_test[:] = minmax_scaler.transform(X_test[:].values)

    test_set = {
        "features": X_test,
        "labels": y_test,
    }

    dump(test_set, "data/processed/test_nn_lin_prob_clus.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
