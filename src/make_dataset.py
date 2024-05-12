from argparse import ArgumentParser, Namespace

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def parse_args() -> Namespace:
    """
    Parse command-line arguments required for the script.

    Returns:
        argparse.Namespace: Parsed arguments with three keys: data_path, save_path, and split_ratio.
    """
    parser = ArgumentParser(
        description="Process and split basketball game data for machine learning use."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input CSV data file."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory path to save output CSV files.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.33,
        help="Test set ratio for splitting data.",
    )
    return parser.parse_args()


def time_convert(time_str: str) -> int:
    """
    Convert time in format 'MM:SS' to total seconds.

    Args:
        time_str (str): Time in 'MM:SS' format.

    Returns:
        int: Time in seconds.
    """
    minutes, seconds = map(int, time_str.split(":"))
    return minutes * 60 + seconds


def main(args: Namespace) -> None:
    """
    Main function that processes the data and saves training and testing datasets.
    
    Args:
        args (argparse.Namespace): Command-line arguments containing data_path, save_path, and split_ratio.
    """
    # Load data.
    df = pd.read_csv(args.data_path)

    # Converting GAME_CLOCK's values from format MM/SS to seconds. The conversion makes the feature numeric.
    df["GAME_CLOCK"] = df.GAME_CLOCK.apply(time_convert)

    # Label encoding attribute LOCATION.
    df["LOCATION"] = LabelEncoder().fit_transform(df["LOCATION"])

    # Removing data points where TOUCH_TIME has negative values.
    df = df[df["TOUCH_TIME"] > 0]

    # Substituting GAME_CLOCK with SHOT_CLOCK.
    df.loc[df.GAME_CLOCK <= 24, "SHOT_CLOCK"] = df.GAME_CLOCK
    df = df.dropna(subset=["SHOT_CLOCK"])

    # Dropping irrelevant features.
    df = df.drop(
        columns=[
            "GAME_ID",
            "MATCHUP",
            "W",
            "PTS_TYPE",
            "CLOSEST_DEFENDER",
            "FGM",
            "PTS",
            "player_name",
        ],
        axis=1,
    )

    # Label encoding the target variable.
    df["SHOT_RESULT"] = LabelEncoder().fit_transform(df["SHOT_RESULT"])

    # Create train and test splits. Save the processed data.
    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(df, test_size=args.split_ratio, random_state=42)
    train.to_csv(args.save_path + "train.csv", index=False)
    test.to_csv(args.save_path + "test.csv", index=False)
    print(
        f"Successfuly proccesed the dataset and saved the splits in {args.save_path}."
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
