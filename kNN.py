import argparse
from statistics import mean
import numpy as np
import pandas as pd

from utils.read_data import read_data
from utils.metrics import metrics
from utils.Add_feat import Add_feat

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

parser = argparse.ArgumentParser()
parser.add_argument('--M', default=4, help='M-fold cross validation')
parser.add_argument('--k', default=7, help='k for kNN')
parser.add_argument('--use_SMOTE', action='store_true')
parser.add_argument ('--feat_reduction', action='store_true', help='drop four least contributing features')
parser.add_argument ('--extra_feat', action='store_true', help='Create extra features utilizing Date')
parser.add_argument('--plot_title', default='', help='title for cf_matrix plot')
args = parser.parse_args()

def main():
    X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
    X_test, y_test = read_data('datasets/algerian_fires_test.csv')
    model = KNeighborsClassifier(n_neighbors=int(args.k))
    scaler = StandardScaler()
    sm = SMOTE(random_state=42)
    if not args.extra_feat:
        # drop first column ("Date" feature)
        X_tr, X_test = X_tr.iloc[:,1:], X_test.iloc[:,1:]
        if args.feat_reduction:
            X_tr = X_tr.drop(columns=['ISI','DMC'])
            X_test = X_test.drop(columns=['ISI','DMC'])
        F1_result, Acc_result = [0]*int(args.M), [0]*int(args.M)
        for m in range(int(args.M)):
            X_val, y_val = X_tr.iloc[46*m:46*(m+1)], y_tr.iloc[46*m:46*(m+1)]
            if m == 0: X_tr_prime, y_tr_prime = X_tr.iloc[46:], y_tr.iloc[46:]
            elif m == 1: 
                X_tr_prime = pd.concat([X_tr.iloc[:46], X_tr.iloc[92:]])
                y_tr_prime = pd.concat([y_tr.iloc[:46], y_tr.iloc[92:]])
            elif m == 2: 
                X_tr_prime = pd.concat([X_tr.iloc[:92], X_tr.iloc[138:]])
                y_tr_prime = pd.concat([y_tr.iloc[:92], y_tr.iloc[138:]])
            else: X_tr_prime, y_tr_prime = X_tr.iloc[:138], y_tr.iloc[:138]
            if args.use_SMOTE:
                X_tr_prime, y_tr_prime = sm.fit_resample(X_tr_prime, y_tr_prime)
            X_tr_prime = scaler.fit_transform(X_tr_prime)
            X_val = scaler.transform(X_val)
            model.fit(X_tr_prime, y_tr_prime)
            y_val_pred = model.predict(X_val)
            F1_result[m], Acc_result[m] = metrics(y_val, y_val_pred, "kNN", work='val')

        print("Val F1_score=", mean(F1_result), "Val Accuracy=", mean(Acc_result))
        print("Training with full dataset!")
        if args.use_SMOTE:
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        X_tr = scaler.fit_transform(X_tr)
        X_test = scaler.transform(X_test)
        model.fit(X_tr, y_tr)
        y_test_pred = model.predict(X_test)
        F1_score, Accuracy = metrics(y_test, y_test_pred, args.plot_title+'_k='+str(args.k))
        print("Test F1_score=", F1_score, "Test Accuracy=", Accuracy)
    else:
        X_val, y_val = X_tr.iloc[-46:], y_tr.iloc[-46:]
        X_tr_prime, y_tr_prime = X_tr.iloc[:-46], y_tr.iloc[:-46]
        X_tr_prime, X_val = Add_feat(X_tr_prime, X_val)
        # drop first column ("Date" feature)
        X_tr_prime, X_val = X_tr_prime.iloc[:,1:], X_val.iloc[:,1:]
        y_tr_prime = y_tr_prime[4:-4]
        if args.use_SMOTE:
                X_tr_prime, y_tr_prime = sm.fit_resample(X_tr_prime, y_tr_prime)
        X_tr_prime = scaler.fit_transform(X_tr_prime)
        X_val = scaler.transform(X_val)
        model.fit(X_tr_prime, y_tr_prime)
        y_val_pred = model.predict(X_val)
        F1_score, Accuracy = metrics(y_val, y_val_pred, args.plot_title+'_k='+str(args.k), 'val')
        print("Val F1_score=", F1_score, "Val Accuracy=", Accuracy)

        print("Training with full dataset!")
        X_tr, X_test = Add_feat(X_tr, X_test)
        # drop first column ("Date" feature)
        X_tr, X_test = X_tr.iloc[:,1:], X_test.iloc[:,1:]
        y_tr = y_tr[4:-4]
        if args.use_SMOTE:
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        X_tr = scaler.fit_transform(X_tr)
        X_test = scaler.transform(X_test)
        model.fit(X_tr, y_tr)
        y_test_pred = model.predict(X_test)
        F1_score, Accuracy = metrics(y_test, y_test_pred, args.plot_title+'_k='+str(args.k))
        print("Test F1_score=", F1_score, "Test Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

