from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import yaml
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

save_model_params_to_yml(model_name='random_forest', model_params= grid_search.best_params_, yml_path='machine_learn\configs\model_best_params.yml')