from argparse import ArgumentParser

import pandas as pd
from joblib import dump


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)
    return parser.parse_args()


def main(args):
    # Load csv file.
    test = pd.read_csv(args.test_split)

    # Prepare labels.
    y_test = test["SHOT_RESULT"]

    # Drop label and unused features.
    X_test = test.drop(
        ["CLOSEST_DEFENDER_PLAYER_ID", "player_id", "SHOT_RESULT"], axis=1
    )

    test_set = {
        "features": X_test,
        "labels": y_test,
    }

    dump(test_set, "data/processed/test_ensemble.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
