from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sys
from utils import save_model_params_to_yml

data = pd.read_csv('machine_learn\data\data_after_clean.csv', index_col=[0])
yml_path = 'machine_learn\configs\model_best_params.yml'
columns_list = list(data.columns)
label_column = 'isDefault'
columns_list.remove(label_column)
features = data[columns_list].copy()
labels = data[label_column].copy()
train_percent = 0.01
seed = 1466
cv_number = 3
X = features[0:int(train_percent*len(features))]
y = labels[0:int(train_percent*len(features))]

# decision_tree
decision_tree = DecisionTreeClassifier()
param_grid = {
    'max_depth': [15, 30, 45, 60], 
    'min_samples_split': [100, 200, 400],
    'min_samples_leaf': [100, 200, 400],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)
save_model_params_to_yml(model_name='decision_tree', model_params= grid_search.best_params_, yml_path=yml_path)


# Random forest
rf_classifier = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 25, 35, 40],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)
save_model_params_to_yml(model_name='random_forest', model_params= grid_search.best_params_,yml_path=yml_path)

# XGBoost
XGBoost_model = xgb.XGBClassifier()
param_grid = {
    'max_depth': [20, 40, 60],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2]
}
grid_search = GridSearchCV(estimator=XGBoost_model, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)
save_model_params_to_yml(model_name='XGBoost', model_params= grid_search.best_params_, yml_path=yml_path)


# SVM
SVM = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.5, 1.0]
}
grid_search = GridSearchCV(estimator=SVM, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)
print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)
save_model_params_to_yml(model_name='SVM', model_params= grid_search.best_params_, yml_path=yml_path)