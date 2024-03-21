# dataset-baseline
Just be a baseline of the dataset from '天池'
# 1. 项目结构
## 1.1. 展示文件
Report| Description
-|-
report_data_clean.ipynb | 介绍数据清洗的方法·
report_model.ipynb | 模型调参和最后运行

## 1.2. 核心代码
File| Description
-|-
data_clean.py|数据清洗
model_tuning.py|模型调参
models.py|常见的机器学习算法
deep_learn_models.py|深度学习模型（此处只实现了MLP）
utils.py|工具函数
main.py|跑模型主文件

## 1.3. 配置文件和日志文件
FileDir| FileName |Description
-|-|-
configs|data_clean.yml|数据清洗参数
configs|model_best_params.yml|已知使模型表现最好参数
outputs|best_score.yml|基于最好参数的各个模型的表现
images|categorize_columns_results.jpg|特征分类图
deep_learn_model_save|MLP.pt|MLP模型保存结果
deep_learn_model_save|MLP_structure.png|MLP架构图(基于Netron)
data|train.csv|所用原始数据
data|data_after_clean.csv|清洗后数据


# 2.数据预处理
## 2.1. 查看变量属性
使用utils.categorize_columns函数对所有的变量属性进行判断，将X划分为"连续变量"，“离散变量”，“字符型变量”，对其分别处理

## 2.2.处理Object类型
a.grade完全由subGrade多对一确定，删除grade，并将subGrade进行one-hot编码,但由于F1-G5样本太小，故将F1-F5合并为F，G1-G5合并为G
b.title属性，无意义，删除
c.employmentLenth共十种取值，转为one-hot编码（后续模型不允许列名中含有'<',故改名）
d.earliesCreditLine和issueDate属性属于月份，取值多且有明显的连续属性，用utils.transform_date_to_int_by_order函数，按先后顺序转化为数字

## 2.3. 处理有偏数据
对部分有偏数据进行放缩，需要进行放缩的变量以及其放缩比例保存在configs/data_clean.yml中
## 2.4. 处理空值
以众数填充
## 2.5. 数据归一化
a. 对于超出上下边界的数据，以边界值填充
b. 对数据进行Z-score标准化，将数据转化为均值为0，标准差为1的数据，方便模型收敛



# 3. 模型调参
用GridSearchCV进行调参，将最好的参数配置到yml文件中

# 4. 模型表现
用已经配置好的参数，进行整个训练集上的训练并获取测试集上的表现

