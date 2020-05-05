# -*- coding: utf-8 -*-
import pandas as pd
import re
import jieba.posseg as psg
import numpy as np

"""
电商产品评论数据情感分析

背景：
    随着电子商务的迅速发展和网络购物的流行，人们对于网络购物的需求变得越来越高，
并且也给电商企业带来巨大的发展机遇，与此同时，这种需求也推动了更多电商企业的崛起，
引发了激烈的竞争。在这种激烈竞争的背景下，除了提高商品质量、压低价格外，了解更多消费者的心声
对电商企业来说也变得越来越有必要。其中一种非常重要的方式就是对消费者的评论文本数据进行内在信息
的分析。
    评论信息中蕴含着消费者对特定产品和服务的主观感受，反映了人们的态度、立场和意见具有非常
宝贵的研究价值。一方面，对企业家来说，企业需要根据海量的评论文本数据更好地了解用户的个人喜好，
从而提高产品质量，改善服务，获取市场上的竞争优势。另一方面，消费者需要在没有看到真正的产品实体
做出购买决策之前，根据其他购物者的评论了解产品的质量，性价比等信息，为购物提供参考依据。

需求：
    1. 对京东商城中美的电热水器的评论进行情感分析
    2. 从评论文本中挖掘用户的需求、意见、购买原因及产品的优缺点
    3. 根据模型结果给出改善产品的建议

实现步骤：
    1. 利用pyhton对京东商城中美的的电热水器的评论进行爬取。
    2. 利用python爬取的京东商城中美的电热水器的评论数据，对评论文本数据进行数据清洗、
    分词、停用词过滤等操作
    3. 对预处理后的数据进行情感分析，将评论文本数据按照情感倾向分为正面评论数据(好评)和
    负面评论数据(差评)
    4. 分别对正、负面评论数据进行LDA主题分析，从对应的结果分析文本评论数据中有价值的内容
"""

# 1 评论去重的代码

# 去重，去除完全重复的数据
reviews = pd.read_csv("../tmp/reviews.csv")
reviews = reviews[['content', 'content_type']].drop_duplicates()
content = reviews['content']



# 2 数据清洗

# 去除去除英文、数字等
# 由于评论主要为京东美的电热水器的评论，因此去除这些词语
strinfo = re.compile('[0-9a-zA-Z]|京东|美的|电热水器|热水器|')
content = content.apply(lambda x: strinfo.sub('', x))

# 3 分词、词性标注、去除停用词代码

# 分词
worker = lambda s: [(x.word, x.flag) for x in psg.cut(s)] # 自定义简单分词函数
seg_word = content.apply(worker)

# 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
n_word = seg_word.apply(lambda x: len(x))  # 每一评论中词的个数
n_content = [[x+1]*y for x,y in zip(list(seg_word.index), list(n_word))]
index_content = sum(n_content, [])  # 将嵌套的列表展开，作为词所在评论的id

seg_word = sum(seg_word, [])
word = [x[0] for x in seg_word]  # 词

nature = [x[1] for x in seg_word]  # 词性

content_type = [[x]*y for x,y in zip(list(reviews['content_type']), list(n_word))]
content_type = sum(content_type, [])  # 评论类型

result = pd.DataFrame({"index_content":index_content, 
                       "word":word,
                       "nature":nature,
                       "content_type":content_type}) 
print(result)
# 删除标点符号
result = result[result['nature'] != 'x']  # x表示标点符号

# 删除停用词
stop_path = open("../data/stoplist.txt", 'r',encoding='UTF-8')
stop = stop_path.readlines()
stop = [x.replace('\n', '') for x in stop]
word = list(set(word) - set(stop))
result = result[result['word'].isin(word)]

# 构造各词在对应评论的位置列
n_word = list(result.groupby(by = ['index_content'])['index_content'].count())
index_word = [list(np.arange(0, y)) for y in n_word]
index_word = sum(index_word, [])  # 表示词语在改评论的位置

# 合并评论id，评论中词的id，词，词性，评论类型
result['index_word'] = index_word




# 4 提取含有名词的评论

# 提取含有名词类的评论
ind = result[['n' in x for x in result['nature']]]['index_content'].unique()
result = result[[x in ind for x in result['index_content']]]
# 5 绘制词云

import matplotlib.pyplot as plt
from wordcloud import WordCloud

frequencies = result.groupby(by = ['word'])['word'].count()
frequencies = frequencies.sort_values(ascending = False)
backgroud_Image=plt.imread('../data/pl.jpg')
wordcloud = WordCloud(font_path="G:/workspace/font/STZHONGS.ttf",
                      max_words=100,
                      background_color='white',
                      mask=backgroud_Image)
my_wordcloud = wordcloud.fit_words(frequencies)
plt.imshow(my_wordcloud)
plt.axis('off') 
plt.show()

# 将结果写出
result.to_csv("../tmp/word.csv", index = False, encoding = 'utf-8')


