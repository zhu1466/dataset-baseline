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
                y_test: pd.DataFrame,
                model_best_configs: dict)->dict:
    '''
    支持向量机（SVM）模型， 最佳模型参数基于model_tuning的结果，保存于configs//model_best_params.yml文件夹下
    '''
    try:
        C = model_best_configs['SVM']['C']
        kernel = model_best_configs['SVM']['kernel']
        gamma = model_best_configs['SVM']['gamma']
        degree = model_best_configs['SVM']['degree']
        coef0 = model_best_configs['SVM']['coef0']

    except:
        print('Model params havent be saved as yml files, please run files from model_tuning first!')
        return False
    svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, probability=True)
    svm.fit(X_train, y_train)

    y_pred_proba = svm.predict_proba(X_test)
    y_pred_classifier = svm.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    acc = accuracy_score(y_test, y_pred_classifier)
    precision = precision_score(y_test, y_pred_classifier)
    recall = recall_score(y_test, y_pred_classifier)
    f1 = f1_score(y_test, y_pred_classifier)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='SVM', model_results= results)
    return results

def random_forest_model(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.DataFrame,
                        y_test: pd.DataFrame,
                        model_best_configs: dict)->dict:
    
    '''
    随机森林模型， 最佳模型参数基于model_tuning的结果，保存于configs//model_best_params.yml文件夹下
    '''
    try:
        n_estimators = model_best_configs['random_forest']['n_estimators']
        min_samples_split = model_best_configs['random_forest']['min_samples_split']
        max_depth = model_best_configs['random_forest']['max_depth']
    except:
        print('Model params havent be saved as yml files, please run files from model_tuning first!')
        return False
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
    rf.fit(X_train, y_train)

    y_pred_proba = rf.predict_proba(X_test)
    y_pred_classifier = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    acc = accuracy_score(y_test, y_pred_classifier)
    precision = precision_score(y_test, y_pred_classifier)
    recall = recall_score(y_test, y_pred_classifier)
    f1 = f1_score(y_test, y_pred_classifier)

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
                    y_test: pd.DataFrame,
                    model_best_configs: dict)->dict:
    '''
    XGBoost模型， 最佳模型参数基于model_tuning的结果，保存于configs//model_best_params.yml文件夹下
    '''
    try:
        max_depth = model_best_configs['XGBoost']['max_depth']
        learning_rate = model_best_configs['XGBoost']['learning_rate']
        n_estimators = model_best_configs['XGBoost']['n_estimators']
        gamma = model_best_configs['XGBoost']['gamma']
    except:
        print('Model params havent be saved as yml files, please run files from model_tuning first!')
        return False
    
    X_train.columns = X_train.columns.str.replace('<', 'beyond')
    X_test.columns = X_test.columns.str.replace('<', 'beyond')

    xgboost_classifier = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, gamma=gamma)
    xgboost_classifier.fit(X_train, y_train)

    y_pred_proba = xgboost_classifier.predict_proba(X_test)
    y_pred_classifier = xgboost_classifier.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    acc = accuracy_score(y_test, y_pred_classifier)
    precision = precision_score(y_test, y_pred_classifier)
    recall = recall_score(y_test, y_pred_classifier)
    f1 = f1_score(y_test, y_pred_classifier)


    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='XGBoost', model_results=results)
    return results

def decision_tree_model( X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame,
                    model_best_configs: dict)->dict:
    '''
    决策树模型， 最佳模型参数基于model_tuning的结果，保存于configs//model_best_params.yml文件夹下
    '''
    try:
        max_depth = model_best_configs['decision_tree']['max_depth']
        min_samples_split = model_best_configs['decision_tree']['min_samples_split']
        min_samples_leaf = model_best_configs['decision_tree']['min_samples_leaf']
        criterion = model_best_configs['decision_tree']['criterion']
        max_features = model_best_configs['decision_tree']['max_features']

    except:
        print('Model params havent be saved as yml files, please run files from model_tuning first!')
        return False
    
    decision_tree_classifier = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, max_features=max_features)
    decision_tree_classifier = decision_tree_classifier.fit(X_train, y_train)

    y_pred_proba = decision_tree_classifier.predict_proba(X_test)
    y_pred_classifier = decision_tree_classifier.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    acc = accuracy_score(y_test, y_pred_classifier)
    precision = precision_score(y_test, y_pred_classifier)
    recall = recall_score(y_test, y_pred_classifier)
    f1 = f1_score(y_test, y_pred_classifier)

    results = {'Auc': auc,
               'Acc': acc,
               'Precision': precision,
               'Recall': recall,
               'F1 score': f1}
    print_model_results(model_name='Decision tree', model_results=results)
    return results



