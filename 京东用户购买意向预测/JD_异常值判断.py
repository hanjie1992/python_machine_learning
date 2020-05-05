import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#定义文件名
ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"

"""
数据背景信息

根据官方给出的数据介绍里，可以知道数据可能存在哪些异常信息
    •用户文件◾用户的age存在未知的情况，标记为-1
        ◾用户的sex存在保密情况，标记为2
        ◾后续分析发现，用户注册日期存在系统异常导致在预测日之后的情况，不过目前针对该特征没有想法，所以不作处理
    •商品文件
        ◾属性a1,a2,a3均存在未知情形，标记为-1
    •行为文件
        ◾model_id为点击模块编号，针对用户的行为类型为6时，可能存在空值
"""

"""
空值判断
"""
def check_empty(file_path, file_name):
    df_file = pd.read_csv(file_path)
    print ('Is there any missing value in {0}? {1}'.format(file_name, df_file.isnull().any().any()))

check_empty(USER_FILE, 'User')
check_empty(ACTION_201602_FILE, 'Action 2')
check_empty(ACTION_201603_FILE, 'Action 3')
check_empty(ACTION_201604_FILE, 'Action 4')
check_empty(COMMENT_FILE, 'Comment')
check_empty(PRODUCT_FILE, 'Product')

"""
由上述简单的分析可知，用户表及行为表中均存在空值记录，而评论表和商品表则不存在，
但是结合之前的数据背景分析商品表中存在属性未知的情况，后续也需要针对分析，进一
步的我们看看用户表和行为表中的空值情况
"""
def empty_detail(f_path, f_name):
    df_file = pd.read_csv(f_path)
    print ('empty info in detail of {0}:'.format(f_name))
    print (pd.isnull(df_file).any())

empty_detail(USER_FILE, 'User')
empty_detail(ACTION_201602_FILE, 'Action 2')
empty_detail(ACTION_201603_FILE, 'Action 3')
empty_detail(ACTION_201604_FILE, 'Action 4')

"""


























上面简单的输出了下存在空值的文件中具体哪些列存在空值(True)，结果如下
* User
    * age
    * sex
    * user_reg_tm
* Action
    * model_id
    
接下来具体看看各文件中的空值情况：
"""
def empty_records(f_path, f_name, col_name):
    df_file = pd.read_csv(f_path)
    missing = df_file[col_name].isnull().sum().sum()
    print ('No. of missing {0} in {1} is {2}'.format(col_name, f_name, missing))
    print ('percent: ', missing * 1.0 / df_file.shape[0])

empty_records(USER_FILE, 'User', 'age')
empty_records(USER_FILE, 'User', 'sex')
empty_records(USER_FILE, 'User', 'user_reg_tm')
empty_records(ACTION_201602_FILE, 'Action 2', 'model_id')
empty_records(ACTION_201603_FILE, 'Action 3', 'model_id')
empty_records(ACTION_201604_FILE, 'Action 4', 'model_id')

"""
对比下数据集的记录数：

       文件       文件说明   记录数
1. JData_User.csv 用户数据集 105,321个用户 
2. JData_Comment.csv 商品评论 558,552条记录 
3. JData_Product.csv 预测商品集合 24,187条记录 
4. JData_Action_201602.csv 2月份行为交互记录 11,485,424条记录 
5. JData_Action_201603.csv 3月份行为交互记录 25,916,378条记录 
6. JData_Action_201604.csv 4月份行为交互记录 13,199,934条记录 

两相对比结合前面输出的情况，针对不同数据进行不同处理
•用户文件 
    ◾age,sex:先填充为对应的未知状态（-1|2），后续作为未知状态的值进一步分析和处理
    ◾user_reg_tm:暂时不做处理
•行为文件
    ◾model_id涉及数目接近一半，而且当前针对该特征没有很好的处理方法，待定
"""
user = pd.read_csv(USER_FILE)
user['age'].fillna('-1', inplace=True)
user['sex'].fillna(2, inplace=True)

print (pd.isnull(user).any())

nan_reg_tm = user[user['user_reg_tm'].isnull()]
print (nan_reg_tm)

print (len(user['age'].unique()))
print (len(user['sex'].unique()))
print (len(user['user_lv_cd'].unique()))

prod = pd.read_csv(PRODUCT_FILE)

print (len(prod['a1'].unique()))
print (len(prod['a2'].unique()))
print (len(prod['a3'].unique()))
# print (len(prod['a2'].unique()))
print (len(prod['brand'].unique()))

"""
未知记录

接下来看看各个文件中的未知记录占的比重
"""
print ('No. of unknown age user: {0} and the percent: {1} '.format(user[user['age']=='-1'].shape[0],
                                                                  user[user['age']=='-1'].shape[0]*1.0/user.shape[0]))
print ('No. of unknown sex user: {0} and the percent: {1} '.format(user[user['sex']==2].shape[0],
                                                                  user[user['sex']==2].shape[0]*1.0/user.shape[0]))


def unknown_records(f_path, f_name, col_name):
    df_file = pd.read_csv(f_path)
    missing = df_file[df_file[col_name] == -1].shape[0]
    print
    ('No. of unknown {0} in {1} is {2}'.format(col_name, f_name, missing))
    print
    ('percent: ', missing * 1.0 / df_file.shape[0])


unknown_records(PRODUCT_FILE, 'Product', 'a1')
unknown_records(PRODUCT_FILE, 'Product', 'a2')
unknown_records(PRODUCT_FILE, 'Product', 'a3')
"""
小结一下
    •空值部分对3条用户的sex,age填充为未知值,注册时间不作处理，此外行为数据部分model_id待定: 43.2%,40.7%,39.0%
    •未知值部分，用户age存在部分未知值: 13.7%，sex存在大量保密情况(超过一半) 52.0%
    •商品中各个属性存在部分未知的情况，a1<a3<a2，分别为： 7.0%,16.7%,15.8%
"""

"""
异常值检测
对于任何的分析，在数据预处理的过程中检测数据中的异常值都是非常重要的一步。
异常值的出现会使得把这些值考虑进去后结果出现倾斜。这里有很多关于怎样定义
什么是数据集中的异常值的经验法则。这里我们将使用Tukey的定义异常值的方法：
一个异常阶（outlier step）被定义成1.5倍的四分位距（interquartile range，IQR）。
一个数据点如果某个特征包含在该特征的IQR之外的特征，那么该数据点被认定为异常点。
"""
# 对于每一个特征，找到值异常高或者是异常低的数据点
# for feature in log_data.keys():
#     # TODO：计算给定特征的Q1（数据的25th分位点）
#     Q1 = np.percentile(log_data[feature], 25)
#
#     # TODO：计算给定特征的Q3（数据的75th分位点）
#     Q3 = np.percentile(log_data[feature], 75)
#
#     # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
#     step = 1.5 * (Q3 - Q1)
#
#     # 显示异常点
#     print
#     ("Data points considered outliers for the feature '{}':".format(feature)
#     display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]))