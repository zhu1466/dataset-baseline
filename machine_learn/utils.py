import pandas as pd
from datetime import datetime

def categorize_columns(df:pd.DataFrame,
                       max_discrete_number: int = 200)-> dict:
    '''
    用于判断输入的df的各个列的属性
    :param max_discrete_number: 离散值最多取值，超过这个上限认为数据是连续的
    :return result: 字典格式，不同key对应值不同属性的列名
    '''
    object_columns = []
    discrete_columns = []
    continuous_columns = []

    for column in df.columns:
        if df[column].dtype == 'object':
            object_columns.append(column)
        elif df[column].nunique() <= max_discrete_number:
            discrete_columns.append(column)
        else:
            continuous_columns.append(column)

    result = {
        'object_columns': object_columns,
        'discrete_columns': discrete_columns,
        'continuous_columns': continuous_columns
    }

    return result

def transform_date_to_int_by_order(date_series: pd.Series)->pd.Series:
    '''
    将日期格式转化为数字并保留顺序
    '''
    if isinstance(date_series[0], str):
        try:
            date_series = date_series.apply(lambda x: datetime.strptime(x,"%Y-%m-%d").date())
        except:
            date_series = date_series.apply(lambda x: datetime.strptime(x,"%b-%Y").date())

    all_date = date_series.value_counts().index.values
    replace_dict = {}
    for i in range(len(all_date)):
        replace_dict[all_date[i]] = i
    date_series = date_series.replace(replace_dict)
    return date_series