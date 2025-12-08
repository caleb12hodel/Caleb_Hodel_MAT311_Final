import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train and return a 3-NN classifier."""
    model = RandomForestClassifier(max_depth = 10, max_features = 6)
    model.fit(X_train, y_train)
    return model