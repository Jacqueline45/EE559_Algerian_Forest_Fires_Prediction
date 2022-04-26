from statistics import mean
import pandas as pd

def Add_feat(X_tr, X_test):
    mean_Ws_tr = [0,0,0,0]
    mean_Ws_test = [0,0,0,0]
    for i in range(4, X_tr.shape[0]-4):
        mean_Ws_tr.append(mean(X_tr['Ws'][i-4:i]))

    for i in range(4, X_test.shape[0]):
        mean_Ws_test.append(mean(X_test['Ws'][i-4:i]))

    mean_Ws_tr.extend([0,0,0,0])

    X_test = X_test.reset_index(drop=True)
    mean_Ws_test[0] = sum(X_tr['Ws'].iloc[-4:])/4
    mean_Ws_test[1] = (sum(X_tr['Ws'].iloc[-3:])+X_test['Ws'].iloc[0])/4
    mean_Ws_test[2] = (sum(X_tr['Ws'].iloc[-2:])+sum(X_test['Ws'].iloc[:2]))/4
    mean_Ws_test[3] = (X_tr['Ws'].iloc[X_tr.shape[0]-1]+sum(X_test['Ws'].iloc[:3]))/4
    
    X_tr = X_tr.assign(mean_Ws = mean_Ws_tr)
    X_test = X_test.assign(mean_Ws = mean_Ws_test)
    
    X_tr = X_tr.drop(X_tr.index[:4])
    X_tr = X_tr.drop(X_tr.index[-4:])

    X_tr = X_tr.reset_index(drop=True)

    return X_tr, X_test

    

