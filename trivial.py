import numpy as np
import pandas as pd

from statistics import mean

from utils.read_data import read_data
from utils.metrics import metrics, plot_val_cf_matrix

def predict(prob_0, prob_1, test_len):
    """
    Args:
        prob_0: probability of assigning to class 0  
        prob_1: probability of assigning to class 1 
        test_len: number of total test data pts.     
    Return: prediction array
    """
    return np.random.choice(2, size=test_len, p=[prob_0, prob_1])

def main():
    X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
    _, y_test = read_data('datasets/algerian_fires_test.csv')
    "========Val========="
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
        N = X_tr_prime.shape[0] 
        N0 = y_tr_prime.value_counts().loc[0] 
        N1 = y_tr_prime.value_counts().loc[1] 
        y_val_pred = predict(N0/N, N1/N, y_val.shape)
        F1_result[m], Acc_result[m], TP[m], TN[m], FP[m], FN[m] = metrics(y_val, y_val_pred, "trivial_system", work='val')
    print("Val F1_score=", mean(F1_result), "Val Accuracy=", mean(Acc_result))
    plot_val_cf_matrix(y_val, y_val_pred, "baseline_system", mean(TP), mean(TN), mean(FP), mean(FN))
    "========Test========"
    # number of total training data pts
    N = X_tr.shape[0] # 184
    # number of training data pts with label 0
    N0 = y_tr.value_counts().loc[0] #69
    # number of training data pts with label 1
    N1 = y_tr.value_counts().loc[1] #115
    y_test_pred = predict(N0/N, N1/N, y_test.shape)
    F1_score, Accuracy = metrics(y_test, y_test_pred, "trivial_system")
    print("Test F1_score=", F1_score, "Test Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

