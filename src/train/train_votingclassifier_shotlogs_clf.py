import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from argparse import ArgumentParser
from joblib import dump, load


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("test_size", type=float)
    return parser.parse_args()


def load_data(filename, test_size):
    df = pd.read_csv(filename, index_col=0)
    labels = df["SHOT_RESULT"]

    # Swap columns - last 3 columns are categorical and labels.
    swap = [
        "LOCATION",
        "FINAL_MARGIN",
        "SHOT_NUMBER",
        "PERIOD",
        "GAME_CLOCK",
        "SHOT_CLOCK",
        "DRIBBLES",
        "TOUCH_TIME",
        "SHOT_DIST",
        "CLOSE_DEF_DIST",
        "CLOSEST_DEFENDER_PLAYER_ID",
        "player_id",
        "SHOT_RESULT",
    ]
    df = df.reindex(columns=swap)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size, random_state=11)

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    train[:-3] = minmax_scaler.fit_transform(train[:-3].values)
    test[:-3] = minmax_scaler.transform(test[:-3].values)

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

    # Prepare estimators.
    logreg = load("./models/logreg.joblib")
    randomforest = load("./models/randomforest.joblib")
    xgboost = load("./models/xgboost.joblib")
    estimators = [("lr", logreg), ("rf", randomforest), ("xgb", xgboost)]

    model_soft = VotingClassifier(estimators, voting="soft", n_jobs=-1, verbose=1)
    model_hard = VotingClassifier(estimators, voting="hard", n_jobs=-1, verbose=1)

    # Train the models.
    model_soft.fit(dataset["train"]["features"], dataset["train"]["labels"])
    model_hard.fit(dataset["train"]["features"], dataset["train"]["labels"])

    # Save the models.
    dump(model_soft, "./models/votingclf_soft.joblib")
    dump(model_hard, "./models/votingclf_hard.joblib")

    # Save test data.
    dump(dataset["test"], "./data/test data/votingclf_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
