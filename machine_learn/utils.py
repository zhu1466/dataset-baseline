import pandas as pd
from datetime import datetime
import yaml
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


def print_model_results(model_results: dict,
                        model_name: str =None)-> None:
    '''
    基于model（dict）的输出结果进行输出
    '''
    if model_name != None:
        print(f'Model Name : {model_name},   performance as follows ')
    for i in model_results.keys():
        if isinstance(model_results[i], float) or isinstance(model_results[i], int):
            print(f'\t----Indicator name {i} : \t{str(round(model_results[i],4))}')
        else:
            print(f'\t----Indicator name {i}  type can be viewed by other method')
        
def save_model_params_to_yml(model_name: str,
                             model_params: dict,
                             yml_path: str) -> None:
    '''
    将模型的参数保存到yml文件中, 供模型使用时读取
    '''
    with open(yml_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader, )
    data[model_name] = model_params
    updated_yaml = yaml.dump(data)
    with open(yml_path, 'w', encoding='utf-8') as file:
        file.write(updated_yaml)
