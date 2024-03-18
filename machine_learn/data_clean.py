import pandas as pd
import numpy as np
from utils import transform_date_to_int_by_order

import yaml

def data_clean(data:pd.DataFrame,
               config_path: str)->pd.DataFrame:
    '''
    对该数据集进行预处理
    '''
    # 处理Object类型
    # 1.grade完全由subGrade多对一确定，删除grade，并将subGrade进行one-hot编码,但由于F1-G5样本太小，故将F1-F5合并为F，G1-G5合并为G
    # 2.title属性，无意义，删除
    # 3.employmentLenth共十种取值，转为one-hot编码（后续模型不允许列名中含有'<',故改名）
    # 4.earliesCreditLine和issueDate属性属于月份，取值多且有明显的连续属性，用utils.transform_date_to_int_by_order函数，按先后顺序转化为数字
    data['earliesCreditLine'] = transform_date_to_int_by_order(data['earliesCreditLine'])
    data['issueDate'] = transform_date_to_int_by_order(data['issueDate'])
    employmentLength_dummies = pd.get_dummies(data['employmentLength'])
    employmentLength_dummies.rename(columns={'< 1 year': 'less than 1 year'}, inplace=True)
    employmentLength_dummies.rename(columns={'10+ years': 'more than 10 years'}, inplace=True)

    subGrade_dummies = pd.get_dummies(data['subGrade'])
    subGrade_dummies['sum_F'] = subGrade_dummies.loc[:, subGrade_dummies.columns.str.match('^F\w{1}$')].sum(axis=1)
    subGrade_dummies['sum_G'] = subGrade_dummies.loc[:, subGrade_dummies.columns.str.match('^G\w{1}$')].sum(axis=1)
    subGrade_dummies.drop(subGrade_dummies.columns[subGrade_dummies.columns.str.match('^G\w{1}$')], inplace=True, axis=1)
    subGrade_dummies.drop(subGrade_dummies.columns[subGrade_dummies.columns.str.match('^F\w{1}$')], inplace=True, axis=1)

    data = pd.concat([data, employmentLength_dummies], axis=1)
    data = pd.concat([data, subGrade_dummies], axis=1)
    data.drop(['title', 'employmentLength', 'subGrade', 'grade'], inplace=True, axis=1)

    # 对部分有偏数据进行放缩，需要进行放缩的变量以及其放缩比例保存在configs/data_clean.yml中
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    for feature_name in config_data['power_size']:
        data[feature_name] = np.power(data[feature_name], config_data['power_size'][feature_name])
        
    # 发现20列中有空值，且最多空值比例5%，以众数填充
    data = data.apply(lambda x : x.fillna(x.mode().iloc[0]), axis=0)

    return data
    