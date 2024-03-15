import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree

from utils import print_model_results
def regression_model(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.DataFrame,
                     y_test: pd.DataFrame)->dict:
    '''
    基本的多元线性回归，以各个特征为X，isDefault为Y
    模型AUC 0.7128002944009528 Acc 0.8015416666666667
    回归结果（95%的置信区间下）：
        不显著的变量及其p值：
                id                    0.847032
                postCode              0.071038
                regionCode            0.781538
                ficoRangeLow          0.177703
                ficoRangeHigh         0.180689
                pubRecBankruptcies    0.985320
                applicationType       0.767805
                policyCode            0.263075
                n0                    0.289308
                n7                    0.590016
                n11                   0.570362
                n12                   0.628440
                D1                    0.241276
        显著的变量有：
                'loanAmnt', 'term', 'interestRate', 'installment',
                'employmentTitle', 'homeOwnership', 'annualIncome',
                'verificationStatus', 'issueDate', 'purpose', 'dti',
                'delinquency_2years', 'openAcc', 'pubRec', 'revolBal', 'revolUtil',
                'totalAcc', 'initialListStatus', 'earliesCreditLine', 'n1', 'n2',
                'n3', 'n4', 'n5', 'n6', 'n8', 'n9', 'n10', 'n13', 'n14', '1 year',
                '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years',
                '7 years', '8 years', '9 years', '< 1 year', 'A1', 'A2', 'A3',
                'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4',
                'C5', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1',
                'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'
    '''

    X_train_with_intercept = sm.add_constant(X_train)
    X_test_with_intercept = sm.add_constant(X_test)

    regression_model = sm.OLS(y_train, X_train_with_intercept)
    results = regression_model.fit()
    # 输出各个系数的 t 值和 p 值
    t_values = results.tvalues
    p_values = results.pvalues

    y_pred = results.predict(X_test_with_intercept)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, np.where(y_pred > 0.5, 1, 0))

    results = {'p_value':p_values,
                't_value':t_values,
                'AUC':auc,
                'Acc':acc}

    return results



def SVM_model(  X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame)->dict:
    svm = SVC()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(results)
    return results

def random_forest_model(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.DataFrame,
                        y_test: pd.DataFrame)->dict:
    rf = RandomForestClassifier(n_estimators=100, random_state=1466)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='Random Forest', model_results=results)
    return results


def XGBoost_model(  X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame)->dict:
    max_depth = 10
    xgboost_classifier = XGBClassifier(max_depth = max_depth)
    xgboost_classifier.fit(X_train, y_train)
    y_pred = xgboost_classifier.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='Random Forest', model_results=results)
    return results

def decision_treel( X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame)->dict:
    max_depth = 10
    decision_tree_classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
    decision_tree_classifier = decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)

    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='Random Forest', model_results=results)
    return results



