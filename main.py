import json
from MAT311.load import *
from MAT311.clean_data import *
from src.data.load_data import load_raw_data
from src.visualization.eda import plot_eda
from src.visualization.confusion_matrix import plot_matrix, plot_performance_comparison_from_dicts
from src.models.dumb_model import train_dumb_model
from src.models.knn_model import train_knn_model
from src.models.random_forest_model import train_rf_model

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
#base dumb model
target = "Churn"
base_features = ["Total Spend", "Usage Frequency"]
base_X_train, base_X_test, base_y_train, base_y_test, base_X_submission = load_data(target, base_features)
dumb_model = train_dumb_model(base_X_train, base_y_train)
base_p = dumb_model.predict_proba(base_X_test)
base_v = dumb_model.predict(base_X_test)
base_Scores = evaluate_model(base_y_test, base_v, base_p[:,1])
print(json.dumps(base_Scores, indent=4))
plot_roc_curve(base_y_test, base_p[:,1])
plot_matrix(base_y_test, base_v)

#KNN --------------------------------------------------------------------------------------------
knn_features = ["Support Calls", "Total Spend", "Contract Length", "Age", "Gender", "Usage Frequency"]
KNN_X_train, KNN_X_test, KNN_y_train, KNN_y_test, KNN_X_submission = load_data(target, knn_features)
knn_model = train_knn_model(KNN_X_train, KNN_y_train)
KNN_p = knn_model.predict_proba(KNN_X_test)
KNN_v = knn_model.predict(KNN_X_test)
KNN_Scores = evaluate_model(KNN_y_test, KNN_v, KNN_p[:,1])
print(json.dumps(KNN_Scores, indent=4))
plot_roc_curve(KNN_y_test, KNN_p[:,1])
plot_matrix(KNN_y_test, KNN_v)

# Random Forest---------------------------------------------------------------------------------
rf_features = ["Support Calls", "Contract Length", "Age", "Gender", "Usage Frequency"]
RF_X_train, RF_X_test, RF_y_train, RF_y_test, RF_X_submission = load_data(target, rf_features)
rf_model = train_rf_model(RF_X_train, RF_y_train)
rf_p = rf_model.predict_proba(RF_X_test)
rf_v = rf_model.predict(RF_X_test)
rf_Scores = evaluate_model(RF_y_test, rf_v, rf_p[:,1])
print(json.dumps(rf_Scores, indent=4))
plot_roc_curve(RF_y_test, rf_p[:,1])
plot_matrix(RF_y_test, rf_v)

# Compare---------------------------------------------------------------------------------------
print("---Creating model comparison...")
plot_performance_comparison_from_dicts(base_Scores, KNN_Scores, rf_Scores)
print("The KNN model is the best model with ROC AUC of .909")


