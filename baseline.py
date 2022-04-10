import numpy as np

from utils.read_data import read_data
from utils.metrics import metrics

def compute_mean(data , labels): 
    """Computes the means per class""" 
    class_means = []
    for c in np.unique(labels):
        class_means.append(data[labels==c].mean()) 
    return class_means

def predict(data, means):
    """Predicts using the nearest mean classifier"""
    distances = [] 
    for m in means :
        distances.append(np.linalg.norm(data-m, axis=1)) 
    return np.argmin(distances, axis=0)

def main():
    X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
    X_test, y_test = read_data('datasets/algerian_fires_test.csv')
    class_means = compute_mean(X_tr, y_tr)
    y_test_pred = predict(X_test.iloc[:,1:], class_means)
    F1_score, Accuracy = metrics(y_test, y_test_pred, "baseline_system")
    print("F1_score=", F1_score, "Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

