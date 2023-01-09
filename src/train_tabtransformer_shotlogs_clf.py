import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
    train, valid = train_test_split(train, test_size=args.test_size, random_state=11)

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    train[:-3] = minmax_scaler.fit_transform(train[:-3].values)
    valid[:-3] = minmax_scaler.transform(valid[:-3].values)
    test[:-3] = minmax_scaler.transform(test[:-3].values)

    # Prepare dataset dictionaries.
    dataset["train"] = {
        "features": train.loc[:, train.columns != "SHOT_RESULT"],
        "labels": train["SHOT_RESULT"],
    }
    dataset["valid"] = {
        "features": valid.loc[:, valid.columns != "SHOT_RESULT"],
        "labels": valid["SHOT_RESULT"],
    }
    dataset["test"] = {
        "features": test.loc[:, test.columns != "SHOT_RESULT"],
        "labels": test["SHOT_RESULT"],
    }

    return dataset


def create_metadata(dataset):
    NUMERIC_FEATURE_NAMES = [
        "FINAL_MARGIN",
        "SHOT_NUMBER",
        "PERIOD",
        "GAME_CLOCK",
        "SHOT_CLOCK",
        "DRIBBLES",
        "TOUCH_TIME",
        "SHOT_DIST",
        "CLOSE_DEF_DIST",
    ]

    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "LOCATION": sorted(list(dataset["LOCATION"].unique())),
        "CLOSEST_DEFENDER_PLAYER_ID": sorted(
            list(dataset["CLOSEST_DEFENDER_PLAYER_ID"].unique())
        ),
        "player_id": sorted(list(dataset["player_id"].unique())),
    }
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

    TARGET_FEATURE_NAME = "SHOT_RESULT"
    TARGET_LABELS = [0, 1]

    WEIGHT_COLUMN_NAME = "fnlwgt"
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    COLUMN_DEFAULTS = [
        [0.0]
        if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME]
        else ["NA"]
        for feature_name in dataset.columns.tolist()
    ]


def main(args):
    dataset = load_data(args.filename, args.test_size)

    # Configure hyperparameters.
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 265
    NUM_EPOCHS = 15
    NUM_TRANSFORMER_BLOCKS = 3  # Number of transformer blocks.
    NUM_HEADS = 4  # Number of attention heads.
    EMBEDDING_DIMS = 16  # Embedding dimensions of the categorical features.
    MLP_HIDDEN_UNITS_FACTORS = [
        2,
        1,
    ]  # MLP hidden layer units, as factors of the number of inputs.
    NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.

    # Train the model.
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    history = model.fit(
        dataset["train"]["features"],
        epochs=NUM_EPOCHS,
        validation_data=dataset["valid"]["features"],
    )

    # Save the model.
    model.save("./models/tabtransformer")

    # Save test data.
    dump(dataset["test"], "./data/test data/tabtransformer_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
