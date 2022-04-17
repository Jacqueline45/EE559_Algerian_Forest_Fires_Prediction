import pandas as pd
"""
Separate datasets into two regions
"""

def Sep_region(X_tr, y_tr, X_test):
    X_tr1, X_tr2 = X_tr.iloc[0], X_tr.iloc[1]
    y_tr1, y_tr2 = [y_tr.iloc[0]], [y_tr.iloc[1]]
    X_test1, X_test2 = X_test.iloc[0], X_test.iloc[1]

    for i in range(2,X_tr.shape[0],2):
        X_tr1 = pd.concat([X_tr1,X_tr.iloc[i]], axis=1)
        X_tr2 = pd.concat([X_tr2,X_tr.iloc[i+1]], axis=1)
        y_tr1.append(y_tr.iloc[i])
        y_tr2.append(y_tr.iloc[i+1])

    for i in range(2,X_test.shape[0],2):
        X_test1 = pd.concat([X_test1,X_test.iloc[i]],axis=1)
        X_test2 = pd.concat([X_test2,X_test.iloc[i+1]],axis=1)

    X_tr1, X_tr2 = X_tr1.T, X_tr2.T
    X_test1, X_test2 = X_test1.T, X_test2.T

    X_tr1 = X_tr1.reset_index(drop=True)
    X_tr2 = X_tr2.reset_index(drop=True)

    X_test1 = X_test1.reset_index(drop=True)
    X_test2 = X_test2.reset_index(drop=True)

    return X_tr1, X_tr2, y_tr1, y_tr2, X_test1, X_test2




