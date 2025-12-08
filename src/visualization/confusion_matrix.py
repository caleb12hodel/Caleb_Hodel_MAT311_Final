from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


def plot_matrix(y_test, predictions):
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, 
                                        display_labels=['No Churn', 'Churn'],
                                        cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def plot_performance_comparison_from_dicts(scores_dict1, scores_dict2, scores_dict3, label1='Dummy Model', label2='KNN', label3='Random Forest') -> None:
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    model1_scores = [
        scores_dict1['accuracy'],
        scores_dict1['precision'],
        scores_dict1['recall'],
        scores_dict1['f1'],
        scores_dict1['roc_auc']
    ]
    
    model2_scores = [
        scores_dict2['accuracy'],
        scores_dict2['precision'],
        scores_dict2['recall'],
        scores_dict2['f1'],
        scores_dict2['roc_auc']
    ]
    model3_scores = [
        scores_dict2['accuracy'],
        scores_dict2['precision'],
        scores_dict2['recall'],
        scores_dict2['f1'],
        scores_dict2['roc_auc']
    ]
    
    df = pd.DataFrame({
        'Metric': metrics, 
        label1: model1_scores, 
        label2: model2_scores,
        label3: model3_scores
    })
    
    df.plot(x='Metric', kind='bar', figsize=(10, 6))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()


