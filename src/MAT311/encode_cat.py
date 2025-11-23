import pandas as pd

def check_cols(churn_train:pd.DataFrame, churn_test:pd.DataFrame , features:list):
    new_features = []
    for feature in features:
        if feature == "Support Calls": #Still not general enough but works for this specific example
            churn_test["Support Calls"] = pd.to_numeric(churn_test[feature], errors='coerce')
            churn_train["Support Calls"] = pd.to_numeric(churn_train[feature], errors='coerce')
            new_features.append(feature)
        if not(pd.api.types.is_numeric_dtype(churn_train[feature])):
            churn_train, new = encode_category(churn_train, feature)
            churn_test, new = encode_category(churn_test, feature)
            new_features.extend(new)
        else:
            new_features.append(feature)
    return churn_train, churn_test, new_features


def encode_category(df:pd.DataFrame, column_name:str):
    original_columns = set(df.columns)
    dummies = pd.get_dummies(df, columns = [column_name], drop_first = True, prefix = column_name, dtype = int)
    new_cols = list(set(dummies.columns) - original_columns)
    # print(new_cols)
    return dummies, new_cols



