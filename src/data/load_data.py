import pandas as pd


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv("data/raw/train.csv")


if __name__ == "__main__":
    df = load_raw_data("data/raw/train.csv")
    print(df.head())
