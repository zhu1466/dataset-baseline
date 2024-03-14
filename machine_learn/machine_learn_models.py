import statsmodels.api as sm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import pandas as pd
def regression_model(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.DataFrame,
                     y_test: pd.DataFrame)->dict:

    X_train_with_intercept = sm.add_constant(X_train)
    X_test_with_intercept = sm.add_constant(X_test)

    regression_model = sm.OLS(y_train, X_train_with_intercept)
    results = regression_model.fit()
    # 输出各个系数的 t 值和 p 值
    t_values = results.tvalues
    p_values = results.pvalues
    print("系数的 t 值:\t", t_values)
    print("系数的 p 值:\t", p_values)

    y_pred = results.predict(X_test_with_intercept)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, np.where(y_pred > 0.5, 1, 0))
    print("模型的 AUC 值:\n", auc)
    print("模型的 Acc 值:\n", acc)

    return {'p_value':p_values,
            't_value':t_values,
            'AUC':auc,
            'Acc':acc}