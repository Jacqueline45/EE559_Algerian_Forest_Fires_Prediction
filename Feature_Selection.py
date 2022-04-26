# Sequential Backward Selection
import argparse
from statistics import mean
import numpy as np
import pandas as pd

from utils.read_data import read_data
from utils.metrics import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

parser = argparse.ArgumentParser()
parser.add_argument('--M', default=4, help='M-fold cross validation')
parser.add_argument('--k', default=7, help='k for kNN')
parser.add_argument('--use_SMOTE', action='store_true')
args = parser.parse_args()

def main():
    X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
    X_test, y_test = read_data('datasets/algerian_fires_test.csv')
    # drop first column ("Date" feature)
    X_tr, X_test = X_tr.iloc[:,1:], X_test.iloc[:,1:]
    model = KNeighborsClassifier(n_neighbors=int(args.k))
    scaler = StandardScaler()
    sm = SMOTE(random_state=42)
    while True:
        if X_tr.shape[1] == 1: break
        SBS_res = dict()
        for col in X_tr.columns:
            X_tr_SBS = X_tr.drop(columns=col)
            F1_result, Acc_result = [0]*args.M, [0]*args.M
            for m in range(args.M):
                X_val, y_val = X_tr_SBS.iloc[46*m:46*(m+1)], y_tr.iloc[46*m:46*(m+1)]
                if m == 0: X_tr_prime, y_tr_prime = X_tr_SBS.iloc[46:], y_tr.iloc[46:]
                elif m == 1: 
                    X_tr_prime = pd.concat([X_tr_SBS.iloc[:46], X_tr_SBS.iloc[92:]])
                    y_tr_prime = pd.concat([y_tr.iloc[:46], y_tr.iloc[92:]])
                elif m == 2: 
                    X_tr_prime = pd.concat([X_tr_SBS.iloc[:92], X_tr_SBS.iloc[138:]])
                    y_tr_prime = pd.concat([y_tr.iloc[:92], y_tr.iloc[138:]])
                else: X_tr_prime, y_tr_prime = X_tr_SBS.iloc[:138], y_tr.iloc[:138]
                if args.use_SMOTE:
                    X_tr_prime, y_tr_prime = sm.fit_resample(X_tr_prime, y_tr_prime)
                X_tr_prime = scaler.fit_transform(X_tr_prime)
                X_val = scaler.transform(X_val)
                model.fit(X_tr_prime, y_tr_prime)
                y_val_pred = model.predict(X_val)
                F1_result[m], Acc_result[m] = metrics(y_val, y_val_pred, "kNN", work='val')
            SBS_res[col] = mean(F1_result)+mean(Acc_result)
        SBS_res = sorted(SBS_res.items(), key = lambda kv:(kv[1], kv[0]))
        drop_col = SBS_res[-1][0]
        print("dropped column:", drop_col)
        X_tr = X_tr.drop(columns=drop_col)

if __name__ == '__main__':
    main()

