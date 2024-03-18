import pandas as pd
import math
from datetime import datetime
import yaml
from pyecharts.charts import Sunburst
from pyecharts import options as opts
import matplotlib.pyplot as plt

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
        'object': object_columns,
        'discrete': discrete_columns,
        'continuous': continuous_columns
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


def plot_categorize_columns_results(categorize_columns_results: dict) ->Sunburst:
    plot_data = []
    for categorize_columns in categorize_columns_results.keys():
        categorize = {}
        categorize_children = []
        for columns in categorize_columns_results[categorize_columns]:
            columns_item = {}
            columns_item['name'] = columns
            columns_item['value'] = 1
            categorize_children.append(columns_item)
        categorize['children'] = categorize_children
        categorize['name'] = categorize_columns
        plot_data.append(categorize)
    c = (
        Sunburst(init_opts=opts.InitOpts(width="1000px", height="600px"))
        .add(
            "",
            data_pair=plot_data,
            highlight_policy="ancestor",
            radius=[0, "95%"],
            sort_="null",
            levels=[
                {},
                {
                    "r0": "15%",
                    "r": "35%",
                    "itemStyle": {"borderWidth": 2},
                    "label": {"rotate": "tangential"},
                },
                {"r0": "35%", "r": "70%", "label": {"align": "right"}},
                {
                    "r0": "70%",
                    "r": "72%",
                    "label": {"position": "outside", "padding": 3, "silent": False},
                    "itemStyle": {"borderWidth": 3},
                },
            ],
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="特征分类"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))
    )

    return c

def plot_different_features(data:pd.DataFrame,
                            plot_method: str = 'bar') ->None:
    width = 30
    height = 100/14*math.floor(len(data.columns)/3)
    fig, axes = plt.subplots(ncols=3, nrows=math.ceil(len(data.columns)/3), figsize=(width, height))
    axes = axes.ravel()
    for i, column in enumerate(data.columns):
        if plot_method == 'bar':
            axes[i].bar(data[column].value_counts().index, data[column].value_counts().values)
        if plot_method == 'hist':
            axes[i].hist(data[column], density=False, bins=15)  
        axes[i].set_title(column)
    plt.tight_layout()
    plt.show()
