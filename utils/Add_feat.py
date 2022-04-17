from statistics import mean
import pandas as pd

def Add_feat(X_tr1, X_tr2, X_test1, X_test2):
    mean_Ws1_tr, mean_Ws2_tr = [0,0,0], [0,0,0]
    mean_Ws1_test, mean_Ws2_test = [0,0,0], [0,0,0]
    for i in range(3, X_tr1.shape[0]-3):
        mean_Ws1_tr.append(mean(X_tr1['Ws'][i-3:i]))
        mean_Ws2_tr.append(mean(X_tr2['Ws'][i-3:i]))

    for i in range(3, X_test1.shape[0]):
        mean_Ws1_test.append(mean(X_test1['Ws'][i-3:i]))
        mean_Ws2_test.append(mean(X_test2['Ws'][i-3:i]))

    mean_Ws1_tr.extend([0,0,0])
    mean_Ws2_tr.extend([0,0,0])

    mean_Ws1_test[0] = sum(X_tr1['Ws'][-3:])/3
    mean_Ws2_test[0] = sum(X_tr2['Ws'][-3:])/3
    mean_Ws1_test[1] = (sum(X_tr1['Ws'][-2:])+X_test1['Ws'][0])/3
    mean_Ws2_test[1] = (sum(X_tr2['Ws'][-2:])+X_test2['Ws'][0])/3
    mean_Ws1_test[2] = (X_tr1['Ws'][X_tr1.shape[0]-1]+sum(X_test1['Ws'][:2]))/3
    mean_Ws2_test[2] = (X_tr2['Ws'][X_tr2.shape[0]-1]+sum(X_test2['Ws'][:2]))/3

    X_tr1['mean_Ws'] = mean_Ws1_tr
    X_tr2['mean_Ws'] = mean_Ws2_tr
    X_test1['mean_Ws'] = mean_Ws1_test
    X_test2['mean_Ws'] = mean_Ws2_test

    X_tr1 = X_tr1.drop(X_tr1.index[:3])
    X_tr1 = X_tr1.drop(X_tr1.index[-3:])

    X_tr2 = X_tr2.drop(X_tr2.index[:3])
    X_tr2 = X_tr2.drop(X_tr2.index[-3:])
    
    X_tr1 = X_tr1.reset_index(drop=True)
    X_tr2 = X_tr2.reset_index(drop=True)
    
    X_tr = pd.concat([X_tr1, X_tr2])
    X_test = pd.concat([X_test1, X_test2])

    return X_tr, X_test

    

