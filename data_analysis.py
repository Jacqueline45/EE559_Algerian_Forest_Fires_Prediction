import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from utils.read_data import read_data

def main(): 
    # numeric feature describe
    X_tr, _ = read_data('datasets/algerian_fires_train.csv')
    X_test, _ = read_data('datasets/algerian_fires_test.csv')
    num_tr_data = X_tr.select_dtypes(['int64','float64'])
    num_test_data = X_test.select_dtypes(['int64','float64'])
    print (num_tr_data.describe().transpose())

    # feature KDE
    for i in list(num_tr_data.columns):
        sns.histplot(num_tr_data[i]).set_title(i)
        plt.savefig('feat_KDE/'+i+'.png')

    for i in list(num_test_data.columns):
        sns.histplot(num_test_data[i]).set_title(i)
        plt.savefig('feat_KDE_test/'+i+'.png')

if __name__ == '__main__':
    main()