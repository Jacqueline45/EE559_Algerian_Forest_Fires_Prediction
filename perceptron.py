import argparse, random
import numpy as np
import pandas as pd
from statistics import mean

from utils.read_data import read_data
from utils.metrics import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE

'''
    Using Stochastic GD - variant 1
    Randomly shuffle before each epoch
    Use Margin = 0.5
    Initial weight vector is random i.i.d. between 0.001 & 0.1
    lr = 100/(1000+i), where i is the number of iterations
    Hailting condition: 
        1. All data pts are correctly classified
        2. After 20000 iterations
'''
# TODO: perceptron with margin

parser = argparse.ArgumentParser()
parser.add_argument('--M', default=4, help='M-fold cross validation')
parser.add_argument('--epoch', default=200, help='# epochs trained')
parser.add_argument('--normalization', action='store_true', help='use min-max normalization')
parser.add_argument('--standardization', action='store_true', help='use standardization')
parser.add_argument('--use_SMOTE', action='store_true')
parser.add_argument('--plot_title', default='', help='title for cf_matrix plot')
args = parser.parse_args()

def J_value(data, label, w):
    J = 0
    for i in range(label.shape[0]):
        z = 1 if label.iloc[i] == 1 else -1
        x = data[i]
        L = np.dot(w, x) * z
        if  L <= 0: J += -L
    return J

def predict(data, label, w):
    result = []
    for i in range(label.shape[0]):
        z = 1 if label.iloc[i] == 1 else -1
        x = data[i]
        if  np.dot(w, x) * z > 0:
            result.append(label.iloc[i])
        else: 
            pred = 1 if label.iloc[i]==0 else 1
            result.append(pred)
    return result

def init_train_param(D):
    """
    Args: D: # features  
    """
    w = [] # weight vector
    for k in range(D):
        w.append(random.uniform(0.001, 0.1))
    it = 0 # iteration counter
    lr = 100/(1000+it) #learning rate
    not_l_s = False # not linearly separable
    c_c = 0 # correctly classified
    w_vec, J_vec = [None] * 500, [None] * 500
    return w, it, lr, not_l_s, c_c, w_vec, J_vec

def train(X, y, N, idx, w, it, lr, not_l_s, c_c, w_vec, J_vec):
    """
    Args:
        idx: training dataframe index for shuffling use
        it: iteration counter
        not_l_s: not linearly separable (bool var.); default: False
        c_c: # correctly classified pts; default: 0
        w_vec: list storing weight vectors
        J_vec: list storing loss values     
    Return: weight vector that gives the lowest loss
    """
    for epoch in range(args.epoch):
        # Shuffle at each epoch
        idx = random.sample(list(idx),N) 
        if it >= 20000: 
            not_l_s = True
            break
        for i in idx:
            if c_c == N: 
                w_hat = w
                print("data is linearly separable")
                print("w_hat=", w_hat)
                print("J=", J_value(X, y, w_hat))
            it += 1
            if it >= 9501 and it <= 10000:
                w_vec[it-9501] = w
                if it == 10000: break
            x = X[i]
            z = 1 if y.iloc[i] == 1 else -1
            if np.dot(w, x) * z <= 0:
                w = w + lr * z * x
                if c_c > 0:
                    c_c = 0
            else:
                c_c += 1

    if not_l_s:
        for i,w in enumerate(w_vec):
            J_vec[i] = J_value(X, y, w)

        index = np.argmin(J_vec)
        w_hat = w_vec[index]
        print("w_hat=", w_hat)
        print("J=", J_vec[index])
    
    return w_hat


def main():
    X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
    X_test, y_test = read_data('datasets/algerian_fires_test.csv')
    # drop first column ("Date" feature)
    X_tr, X_test = X_tr.iloc[:,1:], X_test.iloc[:,1:]
    F1_result, Acc_result = [0]*args.M, [0]*args.M
    sm = SMOTE(random_state=42)
    for m in range(args.M):
        X_val, y_val = X_tr.iloc[46*m:46*(m+1)], y_tr.iloc[46*m:46*(m+1)]
        if m == 0: X_tr_prime, y_tr_prime = X_tr.iloc[46:], y_tr.iloc[46:]
        elif m == 1: 
            X_tr_prime = pd.concat([X_tr.iloc[:46], X_tr.iloc[92:]])
            y_tr_prime = pd.concat([y_tr.iloc[:46], y_tr.iloc[92:]])
        elif m == 2: 
            X_tr_prime = pd.concat([X_tr.iloc[:92], X_tr.iloc[138:]])
            y_tr_prime = pd.concat([y_tr.iloc[:92], y_tr.iloc[138:]])
        else: X_tr_prime, y_tr_prime = X_tr.iloc[:138], y_tr.iloc[:138]

        # Shuffle
        N = X_tr_prime.shape[0] 
        idx = np.arange(N)
        D = X_tr_prime.shape[1]
        w, it, lr, not_linearly_separable, correctly_classified, w_vec, J_vec \
                                                                = init_train_param(D)
        if args.use_SMOTE:
            X_tr_prime, y_tr_prime = sm.fit_resample(X_tr_prime, y_tr_prime)
        if args.normalization or args.standardization:
            if args.normalization:
                scaler = MinMaxScaler()
            elif args.standardization:
                scaler = StandardScaler()
            X_tr_prime = scaler.fit_transform(X_tr_prime)
            X_val = scaler.transform(X_val)
        w_hat = train(X_tr_prime, y_tr_prime, N, idx, w, it, lr, \
                    not_linearly_separable, correctly_classified, w_vec, J_vec)
        
        y_val_pred = predict(X_val, y_val, w_hat)
        F1_result[m], Acc_result[m] = metrics(y_val, y_val_pred, "perceptron", work='val')

    print("Val F1_score=", mean(F1_result), "Val Accuracy=", mean(Acc_result))
    print("Training with full dataset!")
    w, it, lr, not_linearly_separable, correctly_classified, w_vec, J_vec \
                                                                = init_train_param(D)
    if args.use_SMOTE:
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    if args.normalization or args.standardization:
        X_tr = scaler.fit_transform(X_tr)
        X_test = scaler.transform(X_test)
    w_hat = train(X_tr, y_tr, N, idx, w, it, lr, \
                    not_linearly_separable, correctly_classified, w_vec, J_vec)
    y_test_pred = predict(X_test, y_test, w_hat)
    F1_score, Accuracy = metrics(y_test, y_test_pred, args.plot_title)
    print("Test F1_score=", F1_score, "Test Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

