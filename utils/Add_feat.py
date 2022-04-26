from statistics import mean
import pandas as pd

def Add_feat(X_tr, X_test):
    mean_ISI_tr = [0,0,0,0]
    mean_ISI_test = [0,0,0,0]
    for i in range(4, X_tr.shape[0]-4):
        mean_ISI_tr.append(mean(X_tr['ISI'][i-4:i]))

    for i in range(4, X_test.shape[0]):
        mean_ISI_test.append(mean(X_test['ISI'][i-4:i]))

    mean_ISI_tr.extend([0,0,0,0])

    X_test = X_test.reset_index(drop=True)
    mean_ISI_test[0] = sum(X_tr['ISI'].iloc[-4:])/4
    mean_ISI_test[1] = (sum(X_tr['ISI'].iloc[-3:])+X_test['ISI'].iloc[0])/4
    mean_ISI_test[2] = (sum(X_tr['ISI'].iloc[-2:])+sum(X_test['ISI'].iloc[:2]))/4
    mean_ISI_test[3] = (X_tr['ISI'].iloc[X_tr.shape[0]-1]+sum(X_test['ISI'].iloc[:3]))/4
    
    X_tr = X_tr.assign(mean_ISI = mean_ISI_tr)
    X_test = X_test.assign(mean_ISI = mean_ISI_test)
    
    X_tr = X_tr.drop(X_tr.index[:4])
    X_tr = X_tr.drop(X_tr.index[-4:])

    X_tr = X_tr.reset_index(drop=True)

    return X_tr, X_test

    

