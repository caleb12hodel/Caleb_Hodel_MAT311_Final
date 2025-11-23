import pandas as pd


def clean_data_impute(df:pd.DataFrame)->pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "String":
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()[0]
            if mode != None:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].dropna()
            
    return df

def clean_data_drop(df:pd.DataFrame):
    return df.dropna()





def clean_date(df, col):
    df[col] = pd.to_datetime('2024-' + df['Last Payment Date'].astype(str), 
                                                    format='%Y-%m-%d', 
                                                    errors='coerce')
