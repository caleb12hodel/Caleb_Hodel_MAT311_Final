import pandas as pd


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv("data/raw/train.csv")


if __name__ == "__main__":
    df = load_dataset("data/raw/card_transdata.csv")
    print(df.head())
