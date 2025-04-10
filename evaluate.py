import matplotlib.pyplot as plt
import seaborn 
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n -------{model_name} model evaluation--------")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()