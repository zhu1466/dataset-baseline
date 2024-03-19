import pandas as pd
import numpy as np
from data_clean import data_clean
from sklearn.model_selection import train_test_split
from models import *
from mlp import *
import yaml
from utils import save_model_score
data = pd.read_csv('machine_learn\data\data_after_clean.csv', index_col=[0])
model_params_path = 'machine_learn\configs\model_best_params.yml'
model_score_save_path = 'machine_learn\\outputs\\best_score.yml'
with open(model_params_path, 'r', encoding='utf-8') as file:
    model_params = yaml.load(file, Loader=yaml.FullLoader)
data = data[0:5000]
columns_list = list(data.columns)
label_column = 'isDefault'
columns_list.remove(label_column)
features = data[columns_list].copy()
labels = data[label_column].copy()
test_size = 0.3
seed = 1466
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=seed)



# moedel_results = decision_tree_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_best_configs=model_params)
# moedel_results = random_forest_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_best_configs=model_params)
# moedel_results = XGBoost_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_best_configs=model_params)
moedel_results = SVM_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_best_configs=model_params)
# moedel_results = mlp_train_test_proc(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_best_configs=model_params)

save_model_score(moedel_results=moedel_results, save_path=model_score_save_path, model_name='SVM')