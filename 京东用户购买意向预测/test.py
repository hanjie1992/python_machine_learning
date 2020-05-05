import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
"""
特征工程

特征

用户基本特征：
•获取基本的用户特征，基于用户本身属性多为类别特征的特点，对age,sex,usr_lv_cd进行独热编码操作，对于用户注册时间暂时不处理

商品基本特征：
•根据商品文件获取基本的特征
•针对属性a1,a2,a3进行独热编码
•商品类别和品牌直接作为特征

评论特征：
•分时间段，
•对评论数进行独热编码

行为特征：
•分时间段
•对行为类别进行独热编码
•分别按照用户-类别行为分组和用户-类别-商品行为分组统计，然后计算
•用户对同类别下其他商品的行为计数
•不同时间累积的行为计数（3,5,7,10,15,21,30

累积用户特征：
•分时间段
•用户不同行为的
•购买转化率
•均值

用户近期行为特征：
•在上面针对用户进行累积特征提取的基础上，分别提取用户近一个月、近三天的特征，然后提取一个月内用户除去最近三天的行为占据一个月的行为的比重

用户对同类别下各种商品的行为:
•用户对各个类别的各项行为操作统计
•用户对各个类别操作行为统计占对所有类别操作行为统计的比重

累积商品特征:
•分时间段
•针对商品的不同行为的
•购买转化率
•均值

类别特征
•分时间段下各个商品类别的
•购买转化率
•均值

"""
test = pd.read_csv('./data/JData_Action_201602.csv')
test[['user_id','sku_id','model_id','type','cate','brand']] = test[['user_id','sku_id','model_id','type','cate','brand']].astype('float32')
test.dtypes
print(test.info())

test = pd.read_csv('data/JData_Action_201602.csv')
#test[['user_id','sku_id','model_id','type','cate','brand']] = test[['user_id','sku_id','model_id','type','cate','brand']].astype('float32')
test.dtypes
print(test.info())

action_1_path = r'./data/JData_Action_201602.csv'
action_2_path = r'./data/JData_Action_201603.csv'
action_3_path = r'./data/JData_Action_201604.csv'
#action_1_path = r'./data/actions1.csv'
#action_2_path = r'./data/actions2.csv'
#action_3_path = r'./data/actions3.csv'
comment_path = r'./data/JData_Comment.csv'
product_path = r'./data/JData_Product.csv'
user_path = r'./data/JData_User.csv'
#user_path = r'data/user.csv'

comment_date = [
    "2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29",
    "2016-03-07", "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04",
    "2016-04-11", "2016-04-15"
]


def get_actions_0():
    action = pd.read_csv(action_1_path)
    return action


def get_actions_1():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    return action


def get_actions_2():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')

    return action


def get_actions_3():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = action[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')

    return action


def get_actions_10():
    reader = pd.read_csv(action_1_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_20():
    reader = pd.read_csv(action_2_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_30():
    reader = pd.read_csv(action_3_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']] = reader[
        ['user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand']].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


# 读取并拼接所有行为记录文件

def get_all_action():
    action_1 = get_actions_1()
    action_2 = get_actions_2()
    action_3 = get_actions_3()
    actions = pd.concat([action_1, action_2, action_3])     # type: # pd.DataFrame
    # actions = pd.concat([action_1, action_2])
    # actions = pd.read_csv(action_path)
    return actions


# 获取某个时间段的行为记录
def get_actions(start_date, end_date, all_actions):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    actions = all_actions[(all_actions.time >= start_date) & (all_actions.time < end_date)].copy()
    return actions


"""
用户特征

用户基本特征
获取基本的用户特征，基于用户本身属性多为类别特征的特点，对age,sex,usr_lv_cd进行独热编码操作，对于用户注册时间暂时不处理
"""

from sklearn import preprocessing

def get_basic_user_feat():
    # 针对年龄的中文字符问题处理，首先是读入的时候编码，填充空值，然后将其数值化，最后独热编码，此外对于sex也进行了数值类型转换
    user = pd.read_csv(user_path, encoding='gbk')
    #user['age'].fillna('-1', inplace=True)
    #user['sex'].fillna(2, inplace=True)
    user.dropna(axis=0, how='any',inplace=True)
    user['sex'] = user['sex'].astype(int)
    user['age'] = user['age'].astype(int)
    le = preprocessing.LabelEncoder()
    age_df = le.fit_transform(user['age'])
    # print list(le.classes_)

    age_df = pd.get_dummies(age_df, prefix='age')
    sex_df = pd.get_dummies(user['sex'], prefix='sex')
    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
    return user

user = pd.read_csv(user_path, encoding='gbk')
user.isnull().any()

user[user.isnull().values==True]

user.dropna(axis=0, how='any',inplace=True)
user.isnull().any()
#user[user.isnull().values==True]

"""
商品特征¶

商品基本特征
根据商品文件获取基本的特征，针对属性a1,a2,a3进行独热编码，商品类别和品牌直接作为特征
"""
def get_basic_product_feat():
    product = pd.read_csv(product_path)
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
    return product

"""
评论特征
    •分时间段
    •对评论数进行独热编码
"""


def get_comments_product_feat(end_date):
    comments = pd.read_csv(comment_path)
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    comments = comments[comments.dt == comment_date_begin]
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
    # 为了防止某个时间段不具备评论数为0的情况（测试集出现过这种情况）
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0
    df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]

    comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
    # del comments['dt']
    # del comments['comment_num']
    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0', 'comment_num_1',
                         'comment_num_2', 'comment_num_3', 'comment_num_4']]
    return comments

train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date = train_end_date.strftime('%Y-%m-%d')
day = 3

start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
start_date = start_date.strftime('%Y-%m-%d')

comments = pd.read_csv(comment_path)
comment_date_end = train_end_date
comment_date_begin = comment_date[0]
for date in reversed(comment_date):
    if date < comment_date_end:
        comment_date_begin = date
        break
comments = comments[comments.dt == comment_date_begin]
df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
for i in range(0, 5):
    if 'comment_num_' + str(i) not in df.columns:
        df['comment_num_' + str(i)] = 0
df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]

comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
# del comments['dt']
# del comments['comment_num']
comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0', 'comment_num_1',
                     'comment_num_2', 'comment_num_3', 'comment_num_4']]
print(comments.head())


"""
行为特征
•分时间段
•对行为类别进行独热编码
•分别按照用户-类别行为分组和用户-类别-商品行为分组统计，然后计算
    ◾用户对同类别下其他商品的行为计数
    ◾针对用户对同类别下目标商品的行为计数与该时间段的行为均值作差
"""
def get_action_feat(start_date, end_date, all_actions, i):
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'sku_id', 'cate','type']]
    # 不同时间累积的行为计数（3,5,7,10,15,21,30）
    df = pd.get_dummies(actions['type'], prefix='action_before_%s' %i)
    before_date = 'action_before_%s' %i
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
    actions = actions.groupby(['user_id', 'sku_id','cate'], as_index=False).sum()
    # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
    user_cate = actions.groupby(['user_id','cate'], as_index=False).sum()
    del user_cate['sku_id']
    del user_cate['type']
    actions = pd.merge(actions, user_cate, how='left', on=['user_id','cate'])
    #本类别下其他商品点击量
    # 前述两种分组含有相同名称的不同行为的计数，系统会自动针对名称调整添加后缀,x,y，所以这里作差统计的是同一类别下其他商品的行为计数
    actions[before_date+'_1.0_y'] = actions[before_date+'_1.0_y'] - actions[before_date+'_1.0_x']
    actions[before_date+'_2.0_y'] = actions[before_date+'_2.0_y'] - actions[before_date+'_2.0_x']
    actions[before_date+'_3.0_y'] = actions[before_date+'_3.0_y'] - actions[before_date+'_3.0_x']
    actions[before_date+'_4.0_y'] = actions[before_date+'_4.0_y'] - actions[before_date+'_4.0_x']
    actions[before_date+'_5.0_y'] = actions[before_date+'_5.0_y'] - actions[before_date+'_5.0_x']
    actions[before_date+'_6.0_y'] = actions[before_date+'_6.0_y'] - actions[before_date+'_6.0_x']
    # 统计用户对不同类别下商品计数与该类别下商品行为计数均值（对时间）的差值
    actions[before_date+'minus_mean_1'] = actions[before_date+'_1.0_x'] - (actions[before_date+'_1.0_x']/i)
    actions[before_date+'minus_mean_2'] = actions[before_date+'_2.0_x'] - (actions[before_date+'_2.0_x']/i)
    actions[before_date+'minus_mean_3'] = actions[before_date+'_3.0_x'] - (actions[before_date+'_3.0_x']/i)
    actions[before_date+'minus_mean_4'] = actions[before_date+'_4.0_x'] - (actions[before_date+'_4.0_x']/i)
    actions[before_date+'minus_mean_5'] = actions[before_date+'_5.0_x'] - (actions[before_date+'_5.0_x']/i)
    actions[before_date+'minus_mean_6'] = actions[before_date+'_6.0_x'] - (actions[before_date+'_6.0_x']/i)
    del actions['type']
    # 保留cate特征
    # del actions['cate']

    return actions
all_actions = get_all_action()
actions = get_actions(start_date, train_end_date, all_actions)
actions = actions[['user_id', 'sku_id', 'cate','type']]
    # 不同时间累积的行为计数（3,5,7,10,15,21,30）
df = pd.get_dummies(actions['type'], prefix='action_before_%s' %3)
before_date = 'action_before_%s' %3
actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
actions = actions.groupby(['user_id', 'sku_id','cate'], as_index=False).sum()
print(actions.head(20))

