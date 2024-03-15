import pandas as pd
import numpy as np
from data_clean import data_clean
from sklearn.model_selection import train_test_split
from models import *
from mlp import *


data = pd.read_csv('machine_learn\data\data_after_clean.csv', index_col=[0])

data = data[0:100000]
columns_list = list(data.columns)
label_column = 'isDefault'
columns_list.remove(label_column)
features = data[columns_list].copy()
labels = data[label_column].copy()
test_size = 0.3
seed = 1466
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)



#results = mlp_train_test_proc(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
results = SVM_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)