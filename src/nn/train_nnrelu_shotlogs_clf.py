from argparse import ArgumentParser

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import L2
from sklearn.preprocessing import MinMaxScaler


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

    # Build neural network.
    model = Sequential()
    model.add(
        Dense(50, activation="relu", input_shape=(10,), kernel_regularizer=L2(0.0001))
    )
    model.add(Dense(1, activation="sigmoid", kernel_regularizer=L2(0.0001)))
    model.compile(loss="binary_crossentropy", optimizer=SGD(0.01))

    # Train the model.
    model.fit(
        dataset["train"]["features"],
        dataset["train"]["labels"],
        epochs=25,
        batch_size=1000,
        verbose=1,
    )

    # Save the model.
    model.save("models/nn_relu.keras")


if __name__ == "__main__":
    args = parse_args()
    main(args)
