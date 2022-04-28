import numpy as np
import pandas as pd

from statistics import mean

from utils.read_data import read_data
from utils.metrics import metrics, plot_val_cf_matrix

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
    "========Val========"
    F1_result, Acc_result, TP, TN, FP, FN = [0]*4, [0]*4, [0]*4, [0]*4, [0]*4, [0]*4
    for m in range(4):
        X_val, y_val = X_tr.iloc[46*m:46*(m+1)], y_tr.iloc[46*m:46*(m+1)]
        if m == 0: X_tr_prime, y_tr_prime = X_tr.iloc[46:], y_tr.iloc[46:]
        elif m == 1: 
            X_tr_prime = pd.concat([X_tr.iloc[:46], X_tr.iloc[92:]])
            y_tr_prime = pd.concat([y_tr.iloc[:46], y_tr.iloc[92:]])
        elif m == 2: 
            X_tr_prime = pd.concat([X_tr.iloc[:92], X_tr.iloc[138:]])
            y_tr_prime = pd.concat([y_tr.iloc[:92], y_tr.iloc[138:]])
        else: X_tr_prime, y_tr_prime = X_tr.iloc[:138], y_tr.iloc[:138]
    
        class_means = compute_mean(X_tr_prime, y_tr_prime)
        y_val_pred = predict(X_val.iloc[:,1:], class_means)
        F1_result[m], Acc_result[m], TP[m], TN[m], FP[m], FN[m] = metrics(y_val, y_val_pred, "baseline_system", work='val')
    print("Val F1_score=", mean(F1_result), "Val Accuracy=", mean(Acc_result))
    plot_val_cf_matrix(y_val, y_val_pred, "baseline_system", mean(TP), mean(TN), mean(FP), mean(FN))
    "========Test========"
    class_means = compute_mean(X_tr, y_tr)
    y_test_pred = predict(X_test.iloc[:,1:], class_means)
    F1_score, Accuracy = metrics(y_test, y_test_pred, "baseline_system", work='test')
    print("Test F1_score=", F1_score, "Test Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

