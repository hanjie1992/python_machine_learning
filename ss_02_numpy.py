import numpy as np
import random
import matplotlib.pyplot as plt


def numpy_base():
    # 创建数组
    # a,b,c内容相同，注意arange和range区别
    # [1 2 3 4 5] [1 2 3 4 5] [1 2 3 4 5]
    a = np.array([1, 2, 3, 4, 5])
    b = np.array(range(1, 6))
    c = np.arange(1, 6)
    print(a, b, c)
    # 数组类型 numpy.ndarray
    # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    print(type(a), type(b), type(c))
    # 数据的类型
    print(a.dtype)  # int32

    # 指定创建的数组的数据类型
    a = np.array([1, 0, 1, 0], dtype=np.bool)
    print(a)  # [ True False  True False]
    # 修改数组的数据类型
    a1 = a.astype(np.int)
    print(a1)
    # 修改浮点型的小数位数
    b = np.array([random.random() for i in range(10)])
    b1 = b.round(2)
    print(b1)

    # 数组的形状
    # 创建一个二维数组
    a = np.array([[3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]])
    print(a)
    # 查看数组的形状,2行6列
    print(a.shape)  # (2, 6)
    # 修改数组的形状,3行4列
    print(a.reshape(3, 4))
    # 把数组转化为一维数据
    b = a.reshape(1, 12)
    print(b)  # [[3 4 5 6 7 8 4 5 6 7 8 9]]
    print(b.flatten())  # [3 4 5 6 7 8 4 5 6 7 8 9]

    # 数组和数的计算
    # 加减乘除会被广播到每个元素上面进行计算
    a = np.arange(12).reshape(2, 6)
    # 加法减法
    print(a + 1)
    # 乘法除法
    print(a * 3)

    # 数组和数组的计算
    a = np.arange(12).reshape(2, 6)
    b = np.arange(12, 24).reshape(2, 6)
    # 数组和数组的加减法
    print(a + b)
    # 数组和数组的乘除法
    print(a * b)
    c = np.arange(12).reshape(3, 4)
    d = np.arange(6).reshape(1, 6)
    e = [[1], [2]]
    # 行列均不同维度的两个数组，不能计算
    # print(a*c)
    print(a * d)
    print(a * e)

    return None


def youtube_video():
    us_file_path = "../data/US_video_data_numbers.csv"
    us_data = np.loadtxt(us_file_path, delimiter=",", dtype="int")
    print(us_data)
    # 取行
    print(us_data[2])
    # 取连续的多行
    print(us_data[2:])
    # 取不连续的多行
    print(us_data[[2, 8, 10]])

    print(us_data[1, :])
    print(us_data[2:, :])
    print(us_data[[2, 10, 3], :])
    # 取列
    print(us_data[:, 0])
    # 取连续的多列
    print(us_data[:, 2:])
    # 取不连续的多列
    print(us_data[:, [0, 2]])
    # 取多行和多列，取第3行到第五行，第2列到第4列的结果
    # 去的是行和列交叉点的位置
    b = us_data[2:5, 1:4]
    print(b)
    # 取多个不相邻的点
    # 选出来的结果是（0，0） （2，1） （2，3）
    c = us_data[[0, 2, 2], [0, 1, 3]]
    print(c)
    return None


def numpy_t():
    """
    numpy中的转置
        转置是一种变换,对于numpy中的数组来说,
        就是在对角线方向交换数据,目的也是为了更方便的去处理数据
        1.transpose()
        2.T
        以上的两种方法都可以实现二维数组的转置的效果,
        大家能够看出来,转置和交换轴的效果一样
    :return:
    """
    t = np.arange(18).reshape(3, 6)
    print(t)
    print(t.transpose())
    print(t.T)


def numpy_index():
    """
    对于刚刚加载出来的数据,我如果只想选择其中的
    某一列(行)我们应该怎么做呢?
    其实操作很简单,和python中列表的操作一样
    :return:
    """
    a = np.arange(12).reshape(3, 4)
    # 取一行
    print(a[1])
    # 取一列
    print(a[:, 2])
    # 取多行
    print(a[1:3])
    # 取多列
    print(a[:, 2:4])
    print(a[[1, 3], :])
    print(a[:, [2, 4]])
    return None


def numpy_modify():
    """
    numpy中数值的修改,布尔索引,三元运算符
    修改行列的值，我们能够很容易的实现，但是如果条件更复杂呢？
    比如我们想要把t中小于10的数字替换为3
    :return:
    """
    t = np.arange(24).reshape(4, 6)
    print(t[:, 2:4])
    t[:, 2:4] = 0
    print(t)
    # 把t中小于10的数字替换为3
    t1 = np.arange(24).reshape(4, 6)
    t1[t1 < 10] = 3
    print(t1)
    # 把t中小于10的数字替换为0，把大于20的替换为20
    t2 = np.arange(24).reshape(4, 6)
    t2 = np.where(t2 < 10, 0, 10)
    print(t2)
    return None


def numpy_clip():
    """
    numpy中的clip(裁剪)
    :return:
    """
    t = np.arange(12).reshape(3, 4)
    t = t.clip(5, 7)
    print(t)
    return None


def numpy_nan():
    """
    numpy中的nan和inf
    :return:
    """
    a = np.nan
    print(type(a))
    b = np.inf
    print(type(b))

    # numpy中的nan的注意点
    # 1.两个nan是不相等的
    print(np.nan == np.nan)
    print(np.nan != np.nan)
    # 2.判断nan的个数
    a = list([1, 2, np.nan])
    a = np.array(a)
    print(np.count_nonzero(a != a))
    # 3.nan数据进行赋值
    # 4.nan和任何数据计算都为nan
    a[np.isnan(a)] = 0
    print(a)
    return None


def numpy_mean():
    """
    ndarry缺失值填充均值
    :return:
    """
    # t中存在nan值，如何操作把其中的nan填充为每一列的均值
    t1 = np.array([[0., 1., 2., 3., 4., 5.],
                   [6., 7., np.nan, 9., 10., 11.],
                   [12., 13., 14., np.nan, 16., 17.],
                   [18., 19., 20., 21., 22., 23.]])
    for i in range(t1.shape[1]):  # 遍历每一列
        temp_col = t1[:, i]  # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有nan
            # 当前一列不为nan的array
            temp_not_nan_col = temp_col[temp_col == temp_col]
            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    print(t1)
    return None


def numpy_youtube_demo():
    """
    英国和美国各自youtube1000的数据结合之前的matplotlib
    绘制出各自的评论数量的直方图
    :return:
    """
    us_file_path = "../data/US_video_data_numbers.csv"
    uk_file_path = "../data/GB_video_data_numbers.csv"

    # t1 = np.loadtxt(us_file_path,delimiter=",",dtype="int",unpack=True)
    t_us = np.loadtxt(us_file_path, delimiter=",", dtype="int")
    # 取评论的数据
    t_us_comments = t_us[:, -1]
    # 选择比5000小的数据
    t_us_comments = t_us_comments[t_us_comments <= 5000]
    print(t_us_comments.max(), t_us_comments.min())

    d = 50
    bin_nums = (t_us_comments.max() - t_us_comments.min()) // d

    # 绘图
    plt.figure(figsize=(20, 8), dpi=80)
    plt.hist(t_us_comments, bin_nums)
    plt.show()
    return None


def numpy_youtube_demo2():
    """
    希望了解英国的youtube中视频的评论数和喜欢数的关系，应该如何绘制改图
    :return:
    """
    us_file_path = "../data/US_video_data_numbers.csv"
    uk_file_path = "../data/GB_video_data_numbers.csv"

    # t1 = np.loadtxt(us_file_path,delimiter=",",dtype="int",unpack=True)
    t_uk = np.loadtxt(uk_file_path, delimiter=",", dtype="int")
    # 选择喜欢书比50万小的数据
    t_uk = t_uk[t_uk[:, 1] <= 500000]
    t_uk_comment = t_uk[:, -1]
    t_uk_like = t_uk[:, 1]

    plt.figure(figsize=(20, 8), dpi=80)
    plt.scatter(t_uk_like, t_uk_comment)
    plt.show()
    return None


def numpy_stack():
    """
    数组的拼接
    :return:
    """
    t1 = np.arange(12).reshape(3, 4)
    t2 = np.arange(12, 24).reshape(3, 4)
    # 竖直拼接
    print(np.vstack((t1, t2)))
    # 水平拼接
    print(np.hstack((t1, t2)))
    return None


def numpy_exchange():
    """
    数组的行列交换
    :return:
    """
    t = np.arange(12, 24).reshape(3, 4)
    # 行交换
    t1 = t
    t1[[1, 2], :] = t1[[2, 1], :]
    print(t1)
    # 列交换
    t2 = t
    t2[:, [0, 2]] = t2[:, [2, 0]]
    print(t2)
    return None

def numpy_country():
    """
    现在希望把之前案例中两个国家的数据方法一起来研究分析，
    同时保留国家的信息（每条数据的国家来源），应该怎么办
    :return:
    """
    us_data = "../data/US_video_data_numbers.csv"
    uk_data = "../data/GB_video_data_numbers.csv"

    # 加载国家数据
    us_data = np.loadtxt(us_data, delimiter=",", dtype=int)
    uk_data = np.loadtxt(uk_data, delimiter=",", dtype=int)

    # 添加国家信息
    # 构造全为0的数据
    zeros_data = np.zeros((us_data.shape[0], 1)).astype(int)
    ones_data = np.ones((uk_data.shape[0], 1)).astype(int)

    # 分别添加一列全为0,1的数组
    us_data = np.hstack((us_data, zeros_data))
    uk_data = np.hstack((uk_data, ones_data))

    # 拼接两组数据
    final_data = np.vstack((us_data, uk_data))
    print(final_data)
    return None

if __name__ == "__main__":
    # numpy_base()
    # youtube_video()
    # numpy_t()
    # numpy_modify()
    # numpy_clip()
    # numpy_nan()
    # numpy_mean()
    # numpy_stack
    # numpy_youtube_demo()
    # numpy_youtube_demo2()
    # numpy_exchange()
    numpy_country()
    pass
