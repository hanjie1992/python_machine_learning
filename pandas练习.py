import string
import pandas as pd
import numpy as np


# 1. Series练习
# 1.1 series创建 Series 一维，带标签数组
# series对象本质有两个数组构成，一个数组构成对象的键（index,索引），一个数组构成对象的值（values）
a = pd.Series(np.arange(10), index=list(string.ascii_uppercase[:10]))
print("创建的t为：\n", a)
print("查看t的数据类型：", type(a))

# 1.2字典创建series，其中字典的键就是series的索引
a = {string.ascii_uppercase[i]: i for i in range(10)}
b = pd.Series(a)
print("一维Series ,b为:\n", b)
# 对b的索引进行重新赋值
c = pd.Series(b, index=list(string.ascii_uppercase[5:15]))
print("一维Series ,c为:\n", c)

# 1.3 Series切片操作
t = pd.Series(np.arange(10), index=list(string.ascii_uppercase[:10]))
print("从第3个元素开始，到第10个元素，步长是2，t为:", t[2:10:2])
print("取第2个元素", t[1])
print("取第3个、第4个、第6个元素", t[[2, 3, 5]])
print("取出t列表中，值大于4的列表：\n", t[t > 4])
# 1.4 series的索引和值
print("取series的索引：", t.index)
print("取series的值：", t.values)

# 2. DataFrame练习
# DataFrame既有行索引也有列索引，行索引叫index,列索引叫columns
# 2.1 DataFrame创建
df = pd.DataFrame(np.arange(12).reshape((3, 4)))
print("DataFrame创建：", df)

# 2.2创建自定义行索引和列索引的DataFrame
df2 = pd.DataFrame(np.arange((12)).reshape(3, 4), index=list(string.ascii_uppercase[:3])
                   , columns=list(string.ascii_uppercase[-4:]))
print("创建自定义行索引和列索引的DataFrame：", df2)

# 2.3 DataFrame属性练习
print("DataFrame的行数和列数：", df2.shape)
print("DataFrame列的数据类型：\n", df2.dtypes)
print("DataFrame数据的为维度：", df2.ndim)
print("DataFrame行索引：", df2.index)
print("DataFrame列索引：", df2.columns)
print("DataFrame的对象值：\n", df2.values)
print("DataFrame取头几行，默认前5行：\n", df2.head(2))
print("DataFrame显示尾几行，默认5行：\n", df2.tail(2))
# print("DataFrame详细信息：",df.info())
print("DataFrame快速查看统计结果,"
      "计数、均值、标准差、最大值等：\n", df2.describe())

# 2.4 pandas读取外部数据
dog_df = pd.read_csv("../data/dogNames2.csv")
print(dog_df[(800 < dog_df["Count_AnimalName"])
             | (dog_df["Count_AnimalName"] < 1000)])

# 2.5 DataFrame排序
dog_sorted_df = dog_df\
        .sort_values(by="Count_AnimalName", ascending=False)
print("前10个最受欢迎的名字：", dog_sorted_df.head(10))
print("前20个名字最受欢迎的名字：", dog_sorted_df[:20])
print("Count_AnimalName这一列的前20：",
        dog_sorted_df[:20]["Count_AnimalName"])

# 2.6 pandas loc()通过标签获取行数据
print(df2)
# 冒号在loc里面是闭合的
print("获取df2中A行W列的数据：",
        df2.loc["A", "W"])
print("获取df2中A行且 W列、Z列的数据：",
            df2.loc["A", ["W", "Z"]])
print("获取df2中A行、C行 且 W列、Z列的数据：",
            df2.loc[["A", "C"], ["W", "Z"]])
print("获取df2中A行以后，且W列、Z列的数据：",
            df2.loc["A":, ["W", "Z"]])
print("获取df2中A行到C行，且W列、Z列的数据：",
            df2.loc["A":"C", ["W", "Z"]])

# 2.7 panddas iloc()通过位置获取行数据
print(df2)
print("获取df2中第2、3行且第3、4列的数据：\n", df2.iloc[1:3, [2, 3]])
print("获取df2中第2、3行且第2、3列的数据：\n", df2.iloc[1:3, 1:3])

df2.loc["A", "Y"] = 100
df2.iloc[1:2, 0:2] = 200
print("赋值更改数据：\n", df2)

# 2.8 pandas之布尔索引
print("使用次数超过800的狗的名字：\n",
      dog_df[dog_df["Count_AnimalName"] > 800])
#
print("使用次数超过700并且名字的字符串的长度大于4的狗的名字：\n",
      dog_df[(dog_df["Row_Labels"].str.len() > 4)
             & (dog_df["Count_AnimalName"] > 700)])

# 3.pandas 缺失值处理
df3 = pd.DataFrame(np.arange(12).reshape(3, 4))
# 人为创造NaN缺失值
df3.iloc[0, 1] = np.nan
df3.iloc[1, 1] = np.nan
df3.iloc[2, 2] = np.nan
print("判断是否为NaN：", pd.isnull(df3))
print("判断是否为NaN：", pd.notnull(df3))
# 处理1：填充数据
df4 = df3.fillna(df3.mean())
print("填充平均值数据", df4)
# 处理2：删除NaN所在的行列
df3.iloc[2, 2] = 4
print(df3)
df5 = df3.dropna(axis=0, how="any", inplace=False)
print("删除NaN所在的行列：", df5)

# 4.pandas常用统计方法
"""
假设现在我们有一组从2006年到2016年1000部最流行的电影数据，
我们想知道这些电影数据中评分的平均分，导演的人数等信息，我们应该怎么获取
"""
movie_df = pd.read_csv("../data/IMDB-Movie-Data.csv")
print(df.head())
print(df.columns)
print("电影平均分为：", movie_df["Rating"].mean())
print("电影时长最大最小值：",
      movie_df["Runtime (Minutes)"].max(),
      movie_df["Runtime (Minutes)"].min())

# 5.pandas 数据合并
# 5.1 join:默认情况下他是把行索引相同的数据合并到一起
# 创建3行4列，数值都为1的DataFrame，
# 行索引为英文字母前3个,列索引为默认值
t1 = pd.DataFrame(np.ones(shape=(3, 4)),
                  index=list(string.ascii_uppercase[:3]))
# 创建2行5列的，数值都为0的DataFrame。
# 行索引为正数前2个英文字母，列索引为倒数后五个英文字母
t2 = pd.DataFrame(np.zeros(shape=(2, 5)),
                  index=list(string.ascii_uppercase[:2]),
                  columns=list(string.ascii_uppercase[-5:]))
print("t1 join t2的值为：\n",t1.join(t2))

#5.2 merge按照指定的列把数据按照一定的方式
# 合并到一起,默认的合并方式inner，交集
# merge outer，并集，NaN补全
# merge left，左边为准，NaN补全
# merge right，右边为准，NaN补全
t3 = pd.DataFrame([[1,1,"a",1],[1,1,"b",1],[1,1,"c",1]],
                  index=["A","B","C"],columns=["M","N","O","P"])
print("t3:\n",t3)
t4 = pd.DataFrame([[0,0,"c",0,0],[0,0,"d",0,0]],
                  index=["A","B"],columns=["V","W","X","Y","Z"])
print("t4：\n",t4)
print("t3.merge(t4)的结果为：\n",t3.merge(t4,left_on="O",right_on="X"))
print("t3.merge(t4),how='inner'的结果为：\n",
      t3.merge(t4,left_on="O",right_on="X",how="inner"))
print("t3.merge(t4),how='outer'的的结果为：\n",
      t3.merge(t4,left_on="O",right_on="X",how="outer"))
print("t3.merge(t4),how='left'的的结果为：\n",
      t3.merge(t4,left_on="O",right_on="X",how="left"))
print("t3.merge(t4),how='right'的的结果为：\n",
      t3.merge(t4,left_on="O",right_on="X",how="right"))

#6 pandas 分组和聚合
"""
现在我们有一组关于全球星巴克店铺的统计数据，如果
我想知道美国的星巴克数量和中国的哪个多，或者我想
知道中国每个省份星巴克的数量的情况，那么应该怎么办？
语法：
grouped = df.groupby(by="columns_name") grouped是一个DataFrameGroupBy对象，是可迭代的。
"""
starbucks_df = pd.read_csv("../data/starbucks_store_worldwide.csv")
#6.1统计中美两国星巴克店铺的数量
grouped = starbucks_df.groupby(by="Country")
country_count = grouped["Brand"].count()
print("美国星巴克店铺数量：",country_count["US"])
print("中国星巴克店铺数量：",country_count["CN"])

#6.2统计中国每个省店铺的数量
china_data = starbucks_df[starbucks_df["Country"]=="CN"]
grouped = china_data.groupby(by="State/Province").count()["Brand"]
print(grouped)

# 6.3统计每个国家每个省店铺的数量，数据按照多个条件进行分组，返回Series
grouped = starbucks_df.groupby(
    by=[starbucks_df["Country"],
        starbucks_df["State/Province"]]).count()["Brand"]
print(grouped)

#7 pandas 时间序列
"""
语法：
pd.date_range(start=None, end=None, periods=None, freq='D')
start和end以及freq配合能够生成start和end范围内以频率freq的一组时间索引
start和periods以及freq配合能够生成从start开始的频率为freq的periods个时间索引
"""
day_time = pd.date_range(start='20200307',end='20200418',freq="D")
print("生成一段时间内的每天日期：\n",day_time)
month_time = pd.date_range(start='20200307',end='20200918',freq="M")
print("生成一段时间内的每月最后一个日历日：\n",month_time)
index = pd.date_range("20200307",periods=10)
print("periods默认按天生成日期：\n",index)
#时间序列应用于DataFrame
df = pd.DataFrame(np.random.rand(10),index=index)
print("时间序列应用于DataFrame：",df)

#8 空气质量问题
"""
现在我们有北上广、深圳、和沈阳5个城市空气质量数据，
请绘制出5个城市的PM2.5随时间的变化情况观察这组数据
中的时间结构，并不是字符串，这个时候我们应该怎么办？
"""
import matplotlib.pyplot as plt

file_path = "../data/BeijingPM20100101_20151231.csv"
df =pd.read_csv(file_path)

#把分开的时间字符串通过periodIndex的方法转化为pandas的时间类型
period = pd.PeriodIndex(year=df["year"],
                        month=df["month"],
                        day=df["day"],
                        hour=df["hour"],
                        freq="H")
df["datetime"]=period
#把datetime设置为索引
df.set_index("datetime",inplace=True)

#进行降维采样
df = df.resample("7D").mean()
data  =df["PM_US Post"]
data_china = df["PM_Nongzhanguan"]

#画图
_x = data.index
_x = [i.strftime("%Y%m%d") for i in _x]
_x_china = [i.strftime("%Y%m%d") for i in data_china.index]
print(len(_x_china),len(_x_china))
_y = data.values
_y_china = data_china.values

plt.figure(figsize=(20,8),dpi=90)
plt.plot(range(len(_x)),_y,label="US_POST",alpha=0.7)
plt.plot(range(len(_x_china)),_y_china,label="CN_POST",alpha=0.8)

plt.xticks(range(0,len(_x_china),10),list(_x_china)[::10],rotation=45)
plt.legend(loc="best")
plt.show()