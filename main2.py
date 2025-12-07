from MAT311.load import *
from MAT311.clean_data import *

from src.data.load_data import load_raw_data
from src.visualization.eda import plot_eda
from src.models.knn_model import train_knn_model

"""
Target goes in the load_data first argument
A list of features goes in the second argument

load_data returns training and test splits based on the features and targets you pass in
    - Data cleaning is done within load_data
    - load raw data is used just for the sake of seeing the original data
"""
###### GENERAL DATA SET INFORMATION AND VISUALS ##########
print("---Loading data...")
raw_data = load_raw_data()
print(f"Raw dataset shape: {raw_data.shape}")

print("---Cleaning data...")
clean_data = clean_data_impute(raw_data)
print(f"Cleaned dataset shape: {clean_data.shape}")

print("---Creating EDA visuals...")
plot_eda(clean_data)

del raw_data, clean_data

###### MODEL SPECIFIC INFORMATION AND VISUALS ###########
#KNN
target = "Churn"
knn_features = ["Support Calls", "Total Spend", "Contract Length", "Age", "Gender", "Usage Frequency"]
KNN_X_train, KNN_X_test, KNN_y_train, KNN_y_test, KNN_X_submission = load_data(target, knn_features)
knn_model = train_knn_model(KNN_X_train, KNN_y_train)

# Random Forest
rf_features = ["Support Calls", "Total Spend", "Contract Length", "Age", "Gender", "Usage Frequency"]
RF_X_train, KNN_X_test, KNN_y_train, KNN_y_test, KNN_X_submission = load_data(target, knn_features)

rf_model = train_rf_model(RF_X_train, RF_y_train)


