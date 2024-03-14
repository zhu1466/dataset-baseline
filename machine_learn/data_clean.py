import pandas as pd
import numpy as np
from utils import transform_date_to_int_by_order


def data_clean(data:pd.DataFrame)->pd.DataFrame:
    '''
    对该数据集进行预处理
    '''
    # 对连续值处理
    data['loanAmnt'] = np.power(data['loanAmnt'], 0.5)
    data['interestRate'] = np.power(data['interestRate'], 0.4)
    data['installment'] = np.power(data['installment'], 0.5)
    data['employmentTitle'] = np.power(data['employmentTitle'], 0.2)
    data['annualIncome'] = np.power(data['annualIncome'], 0.15)
    data['postCode'] = np.power(data['postCode'], 0.5)
    data['dti'] = np.power(data['dti'], 0.16)
    data['revolBal'] = np.power(data['revolBal'], 0.15)
    data['revolUtil'] = np.power(data['revolUtil'], 0.2)

    # 对离散值的处理
    data['earliesCreditLine'] = transform_date_to_int_by_order(data['earliesCreditLine'])
    data['issueDate'] = transform_date_to_int_by_order(data['issueDate'])

    # str类型数据数值化
    employmentLength_dummies = pd.get_dummies(data['employmentLength'])
    subGrade_dummies = pd.get_dummies(data['subGrade'])
    data = pd.concat([data, employmentLength_dummies], axis=1)
    data = pd.concat([data, subGrade_dummies], axis=1)

    # 抛弃部分变量
    data.drop(['title', 'employmentLength', 'subGrade', 'grade'], inplace=True, axis=1)

    # 以众数补全空值
    data = data.apply(lambda x : x.fillna(x.mode().iloc[0]), axis=0)
    
    return data
