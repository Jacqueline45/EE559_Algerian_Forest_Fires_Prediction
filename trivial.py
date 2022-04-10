import numpy as np

from utils.read_data import read_data
from utils.metrics import metrics

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
    # number of total training data pts
    N = X_tr.shape[0] # 184
    # number of training data pts with label 0
    N0 = y_tr.value_counts().loc[0] #69
    # number of training data pts with label 1
    N1 = y_tr.value_counts().loc[1] #115
    y_test_pred = predict(N0/N, N1/N, y_test.shape)
    F1_score, Accuracy = metrics(y_test, y_test_pred, "trivial_system")
    print("F1_score=", F1_score, "Accuracy=", Accuracy)

if __name__ == '__main__':
    main()

