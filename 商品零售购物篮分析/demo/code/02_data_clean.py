# -*- coding: utf-8 -*-

# 代码8-5 数据转换

import pandas as pd
inputfile='../data/GoodsOrder.csv'
data = pd.read_csv(inputfile,encoding = 'gbk')

# 根据id对“Goods”列合并，并使用“，”将各商品隔开
data['Goods'] = data['Goods'].apply(lambda x:','+x)
data = data.groupby('id').sum().reset_index()

# 对合并的商品列转换数据格式
data['Goods'] = data['Goods'].apply(lambda x :[x[1:]])
data_list = list(data['Goods'])

# 分割商品名为每个元素
data_translation = []
for i in data_list:
    p = i[0].split(',')
    data_translation.append(p)
print('数据转换结果的前5个元素：\n', data_translation[0:5])

# 使用关联分析进行关联
# 导包
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
# 进行one-hot编码
te_ary = te.fit(data_translation).transform(data_translation)
print(type(te_ary))
df = pd.DataFrame(te_ary, columns=te.columns_)
# 利用apriori找出频繁项集
freq = apriori(df, min_support=0.02, use_colnames=True)

# 导入关联规则包
from mlxtend.frequent_patterns import association_rules

# 计算关联规则
result = association_rules(freq, metric="confidence", min_threshold=0.35)
# 排序
result.sort_values(by='confidence', ascending=False, axis=0)
print(result)
result.to_excel("./result.xlsx")
