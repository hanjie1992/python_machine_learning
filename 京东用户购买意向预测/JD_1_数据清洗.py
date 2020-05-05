
import numpy
"""
（1）- 数据清洗

主题：京东用户购买意向预测

故事背景：

京东作为中国最大的自营式电商，在保持高速发展的同时，沉淀了数亿的忠实用户，积累了海量的真实数据。
如何从历史数据中找出规律，去预测用户未来的购买需求，让最合适的商品遇见最需要的人，是大数据应用
在精准营销中的关键问题，也是所有电商平台在做智能化升级时所需要的核心技术。 以京东商城真实的用户、
商品和行为数据（脱敏后）为基础，通过数据挖掘的技术和机器学习的算法，构建用户购买商品的预测模型，
输出高潜用户和目标商品的匹配结果，为精准营销提供高质量的目标群体。

目标：使用京东多个品类下商品的历史销售数据，构建算法模型，预测用户在未来5天内，对某个目标品类
下商品的购买意向。

数据集：
•这里涉及到的数据集是京东最新的数据集：
•JData_User.csv 用户数据集 105,321个用户
•JData_Comment.csv 商品评论 558,552条记录
•JData_Product.csv 预测商品集合 24,187条记录
•JData_Action_201602.csv 2月份行为交互记录 11,485,424条记录
•JData_Action_201603.csv 3月份行为交互记录 25,916,378条记录
•JData_Action_201604.csv 4月份行为交互记录 13,199,934条记录

数据挖掘流程：

（一）.数据清洗
1. 数据集完整性验证
2. 数据集中是否存在缺失值
3. 数据集中各特征数值应该如何处理
4. 哪些数据是我们想要的，哪些是可以过滤掉的
5. 将有价值数据信息做成新的数据源
6. 去除无行为交互的商品和用户
7. 去掉浏览量很大而购买量很少的用户(惰性用户或爬虫用户)

（二）.数据理解与分析
1. 掌握各个特征的含义
2. 观察数据有哪些特点，是否可利用来建模
3. 可视化展示便于分析
4. 用户的购买意向是否随着时间等因素变化

（三）.特征提取
1. 基于清洗后的数据集哪些特征是有价值
2. 分别对用户与商品以及其之间构成的行为进行特征提取
3. 行为因素中哪些是核心？如何提取？
4. 瞬时行为特征or累计行为特征？

（四）.模型建立
1. 使用机器学习算法进行预测 
2. 参数设置与调节
3. 数据集切分？
"""

"""
数据集验证
首先检查JData_User中的用户和JData_Action中的用户是否一致
保证行为数据中的所产生的行为均由用户数据中的用户产生（但是可能存在用户在行为数据中无行为）

思路：利用pd.Merge连接sku 和 Action中的sku, 观察Action中的数据是否减少 Example:
"""
import pandas as pd
# test sample
df1 = pd.DataFrame({'sku':['a','a','e','c'],'data':[1,1,2,3]})
df2 = pd.DataFrame({'sku':['a','b','f']})
df3 = pd.DataFrame({'sku':['a','b','d']})
df4 = pd.DataFrame({'sku':['a','b','c','d']})
# print (pd.merge(df2,df1))
# print (pd.merge(df1,df2))
# print (pd.merge(df3,df1))
# print (pd.merge(df4,df1))
#print (pd.merge(df1,df3))

def user_action_check():
    df_user = pd.read_csv('./data/JData_User.csv',encoding='gbk')
    df_sku = df_user.loc[:,'user_id'].to_frame()
    df_month2 = pd.read_csv('./data/JData_Action_201602.csv',encoding='gbk')
    print ('Is action of Feb. from User file? ', len(df_month2) == len(pd.merge(df_sku,df_month2)))
    df_month3 = pd.read_csv('./data/JData_Action_201603.csv',encoding='gbk')
    print ('Is action of Mar. from User file? ', len(df_month3) == len(pd.merge(df_sku,df_month3)))
    df_month4 = pd.read_csv('./data/JData_Action_201604.csv',encoding='gbk')
    print ('Is action of Apr. from User file? ', len(df_month4) == len(pd.merge(df_sku,df_month4)))

# user_action_check()
"""
Is action of Feb. from User file?  True
Is action of Mar. from User file?  True
Is action of Apr. from User file?  True
结论： User数据集中的用户和交互行为数据集中的用户完全一致
根据merge前后的数据量比对，能保证Action中的用户ID是User中的ID的子集
"""

"""
检查是否有重复记录

除去各个数据文件中完全重复的记录,可能解释是重复数据是有意义的，比如用户同时购买多件商品，
同时添加多个数量的商品到购物车等
"""
def deduplicate(filepath, filename, newpath):
    df_file = pd.read_csv(filepath,encoding='gbk')
    before = df_file.shape[0]
    df_file.drop_duplicates(inplace=True)
    after = df_file.shape[0]
    n_dup = before-after
    print ('No. of duplicate records for ' + filename + ' is: ' + str(n_dup))
    if n_dup != 0:
        df_file.to_csv(newpath, index=None)
    else:
        print ('no duplicate records in ' + filename)

# deduplicate('data/JData_Action_201602.csv', 'Feb. action', 'data/JData_Action_201602_dedup.csv')
# deduplicate('data/JData_Action_201603.csv', 'Mar. action', './data/JData_Action_201603_dedup.csv')
# deduplicate('data/JData_Action_201604.csv', 'Feb. action', './data/JData_Action_201604_dedup.csv')
# deduplicate('data/JData_Comment.csv', 'Comment', './data/JData_Comment_dedup.csv')
# deduplicate('data/JData_Product.csv', 'Product', './data/JData_Product_dedup.csv')
# deduplicate('data/JData_User.csv', 'User', './data/JData_User_dedup.csv')


# df_month2 = pd.read_csv('./data/JData_Action_201602.csv',encoding='gbk')
# IsDuplicated = df_month2.duplicated()
# df_d=df_month2[IsDuplicated]
# print(df_d.groupby('type').count())  #发现重复数据大多数都是由于浏览（1），或者点击(6)产生

"""
检查是否存在注册时间在2016年-4月-15号之后的用户
"""
# df_user = pd.read_csv('./data/JData_User.csv',encoding='gbk')
# df_user['user_reg_tm']=pd.to_datetime(df_user['user_reg_tm'])
# user = df_user.loc[df_user.user_reg_tm  >= '2016-4-15']
# print(user)

"""
由于注册时间是京东系统错误造成，如果行为数据中没有在4月15号之后的数据的话，
那么说明这些用户还是正常用户，并不需要删除。
"""
# df_month = pd.read_csv('data\JData_Action_201604.csv')
# df_month['time'] = pd.to_datetime(df_month['time'])
# month = df_month.loc[df_month.time >= '2016-4-16']
# print(month)
"""
结论：说明用户没有异常操作数据，所以这一批用户不删除
"""

"""
行为数据中的user_id为浮点型，进行INT类型转换
"""
# df_month = pd.read_csv('./data/JData_Action_201602.csv',encoding='gbk')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('./data/JData_Action_201602.csv',index=None)
# df_month = pd.read_csv('./data/JData_Action_201603.csv',encoding='gbk')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('./data/JData_Action_201603.csv',index=None)
# df_month = pd.read_csv('./data/JData_Action_201604.csv',encoding='gbk')
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('./data/JData_Action_201604.csv',index=None)

"""
年龄区间的处理
"""
df_user = pd.read_csv('./data/JData_User.csv',encoding='gbk')
def tranAge(x):
    if x == u'15岁以下':
        x='1'
    elif x==u'16-25岁':
        x='2'
    elif x==u'26-35岁':
        x='3'
    elif x==u'36-45岁':
        x='4'
    elif x==u'46-55岁':
        x='5'
    elif x==u'56岁以上':
        x='6'
    return x
df_user['age']=df_user['age'].apply(tranAge)
print (df_user.groupby(df_user['age']).count())
df_user.to_csv('./data/JData_User.csv',index=None)

"""
为了能够进行上述清洗,在此首先构造了简单的用户(user)行为特征和商品(item)行为特征,
对应于两张表user_table和item_table

user_table
    •user_table特征包括:
    •user_id(用户id),age(年龄),sex(性别),
    •user_lv_cd(用户级别),browse_num(浏览数),
    •addcart_num(加购数),delcart_num(删购数),
    •buy_num(购买数),favor_num(收藏数),
    •click_num(点击数),buy_addcart_ratio(购买加购转化率),
    •buy_browse_ratio(购买浏览转化率),
    •buy_click_ratio(购买点击转化率),
    •buy_favor_ratio(购买收藏转化率)
    
item_table特征包括:
    •sku_id(商品id),attr1,attr2,
    •attr3,cate,brand,browse_num,
    •addcart_num,delcart_num,
    •buy_num,favor_num,click_num,
    •buy_addcart_ratio,buy_browse_ratio,
    •buy_click_ratio,buy_favor_ratio,
    •comment_num(评论数),
    •has_bad_comment(是否有差评),
    •bad_comment_rate(差评率)
"""

"""
构建User_table
"""
#定义文件名
ACTION_201602_FILE = "./data/JData_Action_201602.csv"
ACTION_201603_FILE = "./data/JData_Action_201603.csv"
ACTION_201604_FILE = "./data/JData_Action_201604.csv"
COMMENT_FILE = "./data/JData_Comment.csv"
PRODUCT_FILE = "./data/JData_Product.csv"
USER_FILE = "./data/JData_User.csv"
USER_TABLE_FILE = "./data/User_table.csv"
ITEM_TABLE_FILE = "./data/Item_table.csv"

# 导入相关包
import pandas as pd
import numpy as np
from collections import Counter

# 功能函数: 对每一个user分组的数据进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['user_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

"""
由于用户行为数据量较大,一次性读入可能造成内存错误(Memory Error),因而使用
pandas的分块(chunk)读取.
"""

#对action数据进行统计。根据自己调节chunk_size大小
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True,encoding='gbk')
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[["user_id", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # 按user_id分组，对每一组进行统计，as_index 表示无索引形式返回数据
    df_ac = df_ac.groupby(['user_id'], as_index=False).apply(add_type_count)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates('user_id')

    return df_ac


# 将各个action数据的统计量进行聚合
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()
    # 　构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    # 将大于１的转化率字段置为１(100%)
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac

#　从FJData_User表中抽取需要的字段
def get_from_jdata_user():
    df_usr = pd.read_csv(USER_FILE, header=0)
    df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    return df_usr

# user_base = get_from_jdata_user()
# user_behavior = merge_action_data()
#
# # 连接成一张表，类似于SQL的左连接(left join)
# user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')
# # 保存为user_table.csv
# user_behavior.to_csv(USER_TABLE_FILE, index=False)
#
# user_table = pd.read_csv(USER_TABLE_FILE)
# user_table.head()

"""
构建Item_table
"""
#定义文件名
ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"

# 导入相关包
import pandas as pd
import numpy as np
from collections import Counter

# 读取Product中商品
def get_from_jdata_product():
    df_item = pd.read_csv(PRODUCT_FILE, header=0,encoding='gbk')
    return df_item

# 对每一个商品分组进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]

#对action中的数据进行统计
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[["sku_id", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)

    df_ac = df_ac.groupby(['sku_id'], as_index=False).apply(add_type_count)
    # Select unique row
    df_ac = df_ac.drop_duplicates('sku_id')

    return df_ac

# 获取评论中的商品数据,如果存在某一个商品有两个日期的评论，我们取最晚的那一个
def get_from_jdata_comment():
    df_cmt = pd.read_csv(COMMENT_FILE, header=0)
    df_cmt['dt'] = pd.to_datetime(df_cmt['dt'])
    # find latest comment index
    idx = df_cmt.groupby(['sku_id'])['dt'].transform(max) == df_cmt['dt']
    df_cmt = df_cmt[idx]

    return df_cmt[['sku_id', 'comment_num',
                   'has_bad_comment', 'bad_comment_rate']]

def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.groupby(['sku_id'], as_index=False).sum()

    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac


item_base = get_from_jdata_product()
item_behavior = merge_action_data()
item_comment = get_from_jdata_comment()

# SQL: left join
item_behavior = pd.merge(
    item_base, item_behavior, on=['sku_id'], how='left')
item_behavior = pd.merge(
    item_behavior, item_comment, on=['sku_id'], how='left')

item_behavior.to_csv(ITEM_TABLE_FILE, index=False)

item_table = pd.read_csv(ITEM_TABLE_FILE)
item_table.head()


"""
数据清洗
"""
# 用户清洗
import pandas as pd
df_user = pd.read_csv('data/User_table.csv',header=0)
pd.options.display.float_format = '{:,.3f}'.format  #输出格式设置，保留三位小数
df_user.describe()
"""
由上述统计信息发现： 第一行中根据User_id统计发现有105321个用户，发现有3个用户没有
age,sex字段，而且根据浏览、加购、删购、购买等记录却只有105180条记录，说明存在用户
无任何交互记录，因此可以删除上述用户。

删除没有age,sex字段的用户
"""
df_user[df_user['age'].isnull()]
delete_list = df_user[df_user['age'].isnull()].index
df_user.drop(delete_list,axis=0,inplace=True)

"""
删除无交互记录的用户
"""
#删除无交互记录的用户
df_naction = df_user[(df_user['browse_num'].isnull()) & (df_user['addcart_num'].isnull()) & (df_user['delcart_num'].isnull()) & (df_user['buy_num'].isnull()) & (df_user['favor_num'].isnull()) & (df_user['click_num'].isnull())]
df_user.drop(df_naction.index,axis=0,inplace=True)
print (len(df_user))

"""
统计并删除无购买记录的用户
"""
#统计无购买记录的用户
df_bzero = df_user[df_user['buy_num']==0]
#输出购买数为0的总记录数
print (len(df_bzero))

#删除无购买记录的用户
df_user = df_user[df_user['buy_num']!=0]

df_user.describe()

# 删除爬虫及惰性用户
# 由上表所知，浏览购买转换比和点击购买转换比均值为0.018,0.030，因此这里认为浏览购买转换比和点
# 击购买转换比小于0.0005的用户为惰性用户
bindex = df_user[df_user['buy_browse_ratio']<0.0005].index
print (len(bindex))
df_user.drop(bindex,axis=0,inplace=True)

cindex = df_user[df_user['buy_click_ratio']<0.0005].index
print (len(cindex))
df_user.drop(cindex,axis=0,inplace=True)

print(df_user.describe())

"""
最后这29070个用户为最终预测用户数据集
"""