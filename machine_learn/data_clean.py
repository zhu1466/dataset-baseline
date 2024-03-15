import pandas as pd
import numpy as np
from utils import transform_date_to_int_by_order

import yaml

def data_clean(data:pd.DataFrame,
               config_path: str)->pd.DataFrame:
    '''
    对该数据集进行预处理
    '''
    # 对于有偏数据进行放缩
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    for feature_name in config_data['power_size']:
        data[feature_name] = np.power(data[feature_name], config_data['power_size'][feature_name])

    # 对离散值的处理
    data['earliesCreditLine'] = transform_date_to_int_by_order(data['earliesCreditLine'])
    data['issueDate'] = transform_date_to_int_by_order(data['issueDate'])

    # str类型数据数值化
    employmentLength_dummies = pd.get_dummies(data['employmentLength'])
    subGrade_dummies = pd.get_dummies(data['subGrade'])
    data = pd.concat([data, employmentLength_dummies], axis=1)
    data = pd.concat([data, subGrade_dummies], axis=1)

    # 抛弃部分变量
    #   ——tille: 无意义
    data.drop(['title', 'employmentLength', 'subGrade', 'grade'], inplace=True, axis=1)

    # 以众数补全空值
    data = data.apply(lambda x : x.fillna(x.mode().iloc[0]), axis=0)
    
    return data
    