from matplotlib import pyplot as plt
import random
from matplotlib import rcParams


# rcParams["font.sans-serif"] = "SimHei"
def demo1():
    """
    matplotlib基本要点 案例
    :return:
    """
    # 数据在x轴的位置，是一个可迭代的对象
    x = range(2, 26, 2)
    # 数据在y轴的位置，是一个可迭代的对象
    y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]
    # x轴,y轴的数据一起组成了所有要绘制出的坐标
    # 分别是(2,15),(4,13),(6,14.5)....

    plt.plot(x, y)  # 传入x和y，通过plot绘制折线图
    plt.show()  # 在执行程序的时候展示图形


def demo2():
    """
    matplotlib基本要点 案例改进
    1.设置图片大小
    2.保存图片
    :return:
    """
    """
    设置图片大小
    figure 图形图标的意思，在这里指的就是我们画的图
    通过实例化一个figure并且传递参数，能够在后台自动使用该figure实例
    在图形模糊的时候可以传入dpi参数，让图片更加清晰
    """
    plt.figure(figsize=(20, 8), dpi=80)

    # 数据在x轴的位置，是一个可迭代的对象
    x = range(2, 26, 2)
    # 数据在y轴的位置，是一个可迭代的对象
    y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]
    # x轴,y轴的数据一起组成了所有要绘制出的坐标
    # 分别是(2,15),(4,13),(6,14.5)....

    plt.plot(x, y)  # 传入x和y，通过plot绘制折线图
    plt.xticks(x)  # 设置x的刻度
    # plt.xticks(x[::2])# 当刻度太密集时候使用列表的步长(间隔取值)来解决

    # 保存图片，还可以保存为svg矢量图格式，放大不会有锯齿
    plt.savefig("./t1.png")
    plt.show()  # 在执行程序的时候展示图形


def demo_temperature():
    """
    如果列表a表示10点到12点的每一分钟的气温,如何绘制折线图
    观察每分钟气温的变化情况?
    :return:
    """
    plt.figure(figsize=(20, 8), dpi=90)
    x = range(120)
    # 设置随机种子，让不同时候随机得到的结果都一样
    random.seed(10)
    y = [random.uniform(20, 35) for i in range(120)]
    # 绘制折线图,设置颜色，样式
    plt.plot(x, y, color="red", linestyle="--")
    # 设置中文字体
    rcParams["font.sans-serif"] = "SimHei"
    # 设置x轴上的刻度
    _x_ticks = ["10点{0}分".format(i) for i in x if i < 60]
    _x_ticks += ["11点{0}分".format(i - 60) for i in x if i > 60]

    """
    让列表x中的数据和_x_ticks上的数据都上传，最终会在x轴上一一对应显示出来
    两组数据的长度必须一样，否则不能完全覆盖这个轴
    使用列表的切片操作，每隔5个选一个数据进行展示
    为了让字符串不会覆盖，使用rotation选项，让字符串旋转90°显示 
    """
    plt.xticks(x[::5], _x_ticks[::5], rotation=90)
    # 给图像添加描述信息
    # 设置x轴的label
    plt.xlabel("时间")
    # 设置y轴的label
    plt.ylabel("温度(°C)")

    # 设置标题
    plt.title("10点到12点每分钟的时间变化情况")
    plt.show()
    return None


def scatter_graph():
    """
    散点图案例
    :return:
    """
    y_3 = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22,
           22, 22, 23]
    y_10 = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23, 17, 20, 21, 20, 22, 15, 11, 15, 5, 13, 17, 10, 11,
            13, 12, 13, 6]
    x_3 = range(1, 32)
    x_10 = range(51, 82)
    # 设置字体
    rcParams["font.sans-serif"] = "SimHei"
    # 设置图形大小
    plt.figure(figsize=(20, 8), dpi=90)
    # 使用scatter方法绘制散点图，和之前绘制折线图的唯一区别
    plt.scatter(x_3, y_3, label="3月份")
    plt.scatter(x_10, y_10, label="10月份")
    # 调整x轴的刻度
    _x = list(x_3) + list(x_10)
    _xtick_labels = ["3月{}日".format(i) for i in x_3]
    _xtick_labels += ["4月{}日".format(i - 50) for i in x_10]
    plt.xticks(_x[::3], _xtick_labels[::3], rotation=45)
    # 添加图例
    plt.legend(loc="upper left")
    # 添加描述信息
    plt.xlabel("时间")
    plt.ylabel("温度")
    plt.title("标题")
    # 展示
    plt.show()
    return None

def bar_graph():
    a = ["战狼2", "速度与激情8", "功夫瑜伽", "西游伏妖篇", "变形金刚5：最后的骑士",
         "摔跤吧！爸爸", "加勒比海盗5：死无对证", "金刚：骷髅岛", "极限特工：终极回归",
         "生化危机6：终章","乘风破浪", "神偷奶爸3", "智取威虎山", "大闹天竺",
         "金刚狼3：殊死一战", "蜘蛛侠：英雄归来", "悟空传", "银河护卫队2", "情圣", "新木乃伊", ]

    b = [56.01, 26.94, 17.53, 16.49, 15.45, 12.96, 11.8, 11.61, 11.28, 11.12,
         10.49, 10.3, 8.75, 7.55, 7.32, 6.99, 6.88,6.86, 6.58, 6.23]
    # 设置字体
    rcParams["font.sans-serif"] = "SimHei"
    #设置图形大小
    plt.figure(figsize=(20.,15),dpi=90)
    #绘制条形图
    plt.bar(range(len(a)),a,width=0.3,color="orange")
    #设置字符串到x轴
    plt.xticks(range(len(a)),a,rotation=90)

    plt.show()
    return None

def bar_graph_2():
    a = ["猩球崛起3：终极之战", "敦刻尔克", "蜘蛛侠：英雄归来", "战狼2"]
    b_16 = [15746, 312, 4497, 319]
    b_15 = [12357, 156, 2045, 168]
    b_14 = [2358, 399, 2358, 362]
    bar_width = 0.2
    # 设置字体
    rcParams["font.sans-serif"] = "SimHei"
    x_14 = list(range(len(a)))
    x_15 = [i+bar_width for i in x_14]
    x_16 = [i+bar_width*2 for i in x_14]

    #设置图形大小
    plt.figure(figsize=(20,8),dpi=90)

    plt.bar(range(len(a)),b_14,width=bar_width,label="9月14日")
    plt.bar(x_15,b_15,width=bar_width,label="9月15日")
    plt.bar(x_16,b_16,width=bar_width,label="9月16日")

    #设置图例
    plt.legend()
    plt.xticks(x_15,a)
    plt.show()

def hist_graph():
    a = [131, 98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119,
         128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115, 99, 136, 126, 134,
         95, 138, 117, 111, 78, 132, 124, 113, 150, 110, 117, 86, 95, 144, 105, 126, 130, 126,
         130, 126, 116, 123, 106, 112, 138, 123, 86, 101, 99, 136, 123, 117, 119, 105, 137, 123,
         128, 125, 104, 109, 134, 125, 127, 105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102,
         120, 114, 105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134, 156, 106, 117,
         127, 144, 139, 139, 119, 140, 83, 110, 102, 123, 107, 143, 115, 136, 118, 139, 123, 112,
         118, 125, 109, 119, 133, 112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,
         115, 146, 137, 116, 103, 144, 83, 123, 111, 110, 111, 100, 154, 136, 100, 118, 119, 133,
         134, 106, 129, 126, 110, 111, 109, 141, 120, 117, 106, 149, 122, 122, 110, 118, 127, 121,
         114, 125, 126, 114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137, 92, 121, 112,
         146, 97, 137, 105, 98, 117, 112, 81, 97, 139, 113, 134, 106, 144, 110, 137, 137, 111,
         104, 117, 100, 111, 101, 110, 105, 129, 137, 112, 120, 113, 133, 112, 83, 94, 146, 133,
         101, 131, 116, 111, 84, 137, 115, 122, 106, 144, 109, 123, 116, 111, 111, 133, 150]
    #计算组数
    d = 3
    num_bins = (max(a)-min(a))//d
    print(max(a), min(a), max(a) - min(a))
    print(num_bins)
    #设置图形大小
    plt.figure(figsize=(20,8),dpi=80)
    #绘制直方图
    plt.hist(a,num_bins)
    #设置x轴的刻度
    plt.xticks(range(min(a),max(a)+d,d))
    plt.grid()
    plt.show()
    return None

if __name__ == "__main__":
    # demo1()
    # demo2()
    # demo_temperature()
    # scatter_graph()
    # bar_graph()
    # bar_graph_2()
    hist_graph()
    pass

