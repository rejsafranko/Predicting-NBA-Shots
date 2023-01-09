import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import layers
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

    metadata = {
        "NUMERIC_FEATURE_NAMES": NUMERIC_FEATURE_NAMES,
        "CATEGORICAL_FEATURES_WITH_VOCABULARY": CATEGORICAL_FEATURES_WITH_VOCABULARY,
        "CATEGORICAL_FEATURE_NAMES": CATEGORICAL_FEATURE_NAMES,
        "TARGET_FEATURE_NAME": TARGET_FEATURE_NAME,
        "TARGET_LABELS": TARGET_LABELS,
        "WEIGHT_COLUMN_NAME": WEIGHT_COLUMN_NAME,
        "COLUMN_DEFAULTS": COLUMN_DEFAULTS,
    }

    return metadata


def create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


def encode_inputs(
    inputs,
    embedding_dims,
    CATEGORICAL_FEATURE_NAMES,
    CATEGORICAL_FEATURES_WITH_VOCABULARY,
):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):

    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Tansformer.
    encoded_categorical_features = tf.stack(encoded_categorical_feature_list, axis=1)
    # Concatenate numerical features.
    numerical_features = layers.concatenate(numerical_feature_list)

    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(
            column_indices
        )

    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = layers.Add(name=f"skip_connection1_{block_idx}")(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)
        # Skip connection 2.
        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
        # Layer normalization 2.
        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}", epsilon=1e-6
        )(x)

    # Flatten the "contextualized" embeddings of the categorical features.
    categorical_features = layers.Flatten()(encoded_categorical_features)
    # Apply layer normalization to the numerical features.
    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
    # Prepare the input for the final MLP block.
    features = layers.concatenate([categorical_features, numerical_features])

    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    # Add a sigmoid as a binary classifer.
    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def main(args):
    dataset = load_data(args.filename, args.test_size)
    dataset_metadata = create_metadata(dataset["train"])

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

    model = create_tabtransformer_classifier(
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        num_heads=NUM_HEADS,
        embedding_dims=EMBEDDING_DIMS,
        mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
        dropout_rate=DROPOUT_RATE,
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
        batch_size=BATCH_SIZE,
    )

    # Save the model.
    model.save("./models/tabtransformer")

    # Save test data.
    dump(dataset["test"], "./data/test data/tabtransformer_test_data.joblib")


if __name__ == "__main__":
    args = parse_args()
    main(args)
