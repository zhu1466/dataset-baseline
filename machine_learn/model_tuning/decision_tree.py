from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sys
sys.path.append('machine_learn/')
from utils import save_model_params_to_yml

data = pd.read_csv('machine_learn\data\data_after_clean.csv', index_col=[0])

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

rf_classifier = DecisionTreeClassifier()

param_grid = {
    'max_depth': [15, 30, 45, 60], 
    'min_samples_split': [100, 200, 400],
    'min_samples_leaf': [100, 200, 400],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}


grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)

print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)

save_model_params_to_yml(model_name='decision_tree', model_params= grid_search.best_params_, yml_path='machine_learn\configs\model_best_params.yml')