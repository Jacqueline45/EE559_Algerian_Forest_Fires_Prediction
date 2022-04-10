import pandas as pd

def read_data(file):
    """Reads a CSV file and returns a data matrix and the labels separatelly"""
    data = pd.read_csv(file)
    return data.iloc[:, :-1], data.iloc[:,-1]