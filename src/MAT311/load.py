import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
    precision_score, recall_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from MAT311.clean_data import *
from MAT311.encode_cat import check_cols
import os


def scale_data(train:pd.DataFrame, test:pd.DataFrame, test_submission:pd.DataFrame)->list:
    scaler = StandardScaler()
    
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        numeric_cols = test.select_dtypes(include=[np.number]).columns.tolist()

        #Apply Training Scale to both test sets!
        test[numeric_cols] = scaler.transform(test[numeric_cols])
        test_submission[numeric_cols] = scaler.transform(test_submission[numeric_cols])
    return train, test, test_submission

    


def load_data(target:str, features:list = None)->list:
    churn_train = pd.read_csv("./data/raw/train.csv") # add back ../ for notebook work and then re build
    churn_test = pd.read_csv("./data/raw/test.csv")   # add back ../ for notebook work and then re build
    churn_train = churn_train.set_index("CustomerID", drop = False)
    churn_test = churn_test.set_index("CustomerID", drop = False)

    #Make Sure all Features are numerical
    churn_train, churn_test, features = check_cols(churn_train, churn_test , features)

    #Clean Data
    churn_train = clean_data_impute(churn_train)
    churn_test = clean_data_impute(churn_test)
    churn_train.to_csv(f"./data/processed/processed.csv",index= False)


    #Separate X and y
    y = churn_train[target].copy()
    X = churn_train.drop(columns=[target])

    
    #Subset on features if provided
    if features is not None:
        X = X[features]
        X_submission = churn_test[features].copy()
    
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    #Scale Data
    X_train, X_test, X_submission = scale_data(X_train, X_test, X_submission)

    return X_train, X_test, y_train, y_test, X_submission


# def load_data_for_validated_model(target:str, features:list = None)->list:
#     churn_train = pd.read_csv("../data/raw/train.csv")
#     churn_test = pd.read_csv("../data/raw/test.csv")
#     churn_train = churn_train.set_index("CustomerID", drop = False)
#     churn_test = churn_test.set_index("CustomerID", drop = False)
   
#     if features is None:
#         features = churn_test.columns
        

#     churn_train, churn_test, features = check_cols(churn_train, churn_test , features)

#     #Clean Data
#     churn_train = clean_data_impute(churn_train)
#     churn_test = clean_data_impute(churn_test)

#     #Separate X and y
#     y = churn_train[target].copy()
#     X = churn_train.drop(columns=[target])

#     #Subset on features if provided
#     if features is not None:
#         X = X[features]
#         X_submission = churn_test[features].copy()

#     X_train = churn_train[features].copy()
#     X_submission = churn_test[features].copy()  # Don't include CustomerID

#     X_test = X_train.copy() #Dummy these so the scale data function works
#     y_test = y.copy()
#     y_train = y


#     #Scale Data
#     X_train, X_test, X_submission = scale_data(X_train, X_test, X_submission)

#     return X_train, X_test, y_train, y_test, X_submission



def evaluate_model(y_test, predictions, probabilities=None)->dict:
    scores = {}
    scores["accuracy"] = accuracy_score(y_test, predictions)
    
    if probabilities is not None:
        scores["roc_auc"] = roc_auc_score(y_test, probabilities)
    else:
        scores["roc_auc"] = roc_auc_score(y_test, predictions)
    
    scores["precision"] = precision_score(y_test, predictions)
    scores["recall"] = recall_score(y_test, predictions)
    scores["f1"] = f1_score(y_test, predictions)
    
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    scores["true_negatives"] = int(tn)
    scores["false_positives"] = int(fp)
    scores["false_negatives"] = int(fn)
    scores["true_positives"] = int(tp)
    
    return scores


def plot_roc_curve(y_test:np.array, predictions:np.array):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def create_submission(model, X_submission, num) -> None:
    preds = model.predict_proba(X_submission)[:,1]
    results_df = pd.DataFrame({
        "CustomerID": X_submission.index,
        "Churn": preds
    })
    results_df.to_csv(f"../submission/submission{num}.csv",index= False)
    return results_df

    


