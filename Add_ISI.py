from cv2 import fastNlMeansDenoisingColored
import pandas as pd

from utils.read_data import read_data
from utils.Add_feat import Add_feat

X_tr, y_tr = read_data('datasets/algerian_fires_train.csv')
X_test, y_test = read_data('datasets/algerian_fires_test.csv')
X_tr, X_test = Add_feat(X_tr, X_test)
y_tr = y_tr[4:-4]

data_tr = pd.concat([X_tr, y_tr], axis=1)
data_test = pd.concat([X_test, y_test], axis=1)

data_tr.to_csv('datasets/train_add_ISI.csv', index=False)
data_test.to_csv('datasets/test_add_ISI.csv', index=False)