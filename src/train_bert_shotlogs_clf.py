import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
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

    # Keep only numeric features.
    features = df.drop(["SHOT_RESULT"], axis=1)

    # Join features and labels.
    df = pd.concat([features, labels], axis=1)

    # Create dataset split.
    dataset = dict()
    train, test = train_test_split(df, test_size=test_size, random_state=2311)

    # Normalize features.
    minmax_scaler = MinMaxScaler()
    train[:-1] = minmax_scaler.fit_transform(train[:-1].values)
    test[:-1] = minmax_scaler.transform(test[:-1].values)

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

    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    # Preprocess the data
    X_bert = [
        tokenizer.encode(str(x), add_special_tokens=True)
        for x in dataset["train"]["features"]
    ]
    features_bert = tf.keras.preprocessing.sequence.pad_sequences(
        features_bert, maxlen=512, dtype="long", truncating="post", padding="post"
    )

    # Define a loss function and an optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Use the Adam optimizer to minimize the loss
    for epoch in range(5):
        with tf.GradientTape() as tape:
            logits = model(X_bert, training=True)
            loss_value = loss_fn(dataset["train"]["labels"], logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Make predictions on test data
    # logits = model(X_test, training=False)
    # predictions = tf.argmax(logits, axis=-1).numpy()

    # Save the model.
    model.save("./models/bert")

    # Save test data.
    dump(dataset["test"], "./data/test data/nnrelu_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
