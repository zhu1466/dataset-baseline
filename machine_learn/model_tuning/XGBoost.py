import xgboost as xgb
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

rf_classifier = xgb.XGBClassifier()

param_grid = {
    'max_depth': [20, 40, 60],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2]
}


grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc', cv=cv_number)
grid_search.fit(X, y)

print("Best Parameters: ", grid_search.best_params_)
print("Best AUC Score: ", grid_search.best_score_)

save_model_params_to_yml(model_name='XGBoost', model_params= grid_search.best_params_, yml_path='machine_learn\configs\model_best_params.yml')