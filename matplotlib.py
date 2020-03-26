from matplotlib import pyplot as plt
from matplotlib import rcParams
# rcParams["font.sans-serif"] = "SimHei"
def demo1():
    """
    matplotlib基本要点 案例
    :return:
    """
    # 数据在x轴的位置，是一个可迭代的对象
    x = range(2,26,2)
    # 数据在y轴的位置，是一个可迭代的对象
    y = [15,13,14.5,17,20,25,26,26,27,22,18,15]
    # x轴,y轴的数据一起组成了所有要绘制出的坐标
    # 分别是(2,15),(4,13),(6,14.5)....

    plt.plot(x,y) # 传入x和y，通过plot绘制折线图
    plt.show() # 在执行程序的时候展示图形

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
    x = range(2,26,2)
    # 数据在y轴的位置，是一个可迭代的对象
    y = [15,13,14.5,17,20,25,26,26,27,22,18,15]
    # x轴,y轴的数据一起组成了所有要绘制出的坐标
    # 分别是(2,15),(4,13),(6,14.5)....

    plt.plot(x,y) # 传入x和y，通过plot绘制折线图
    plt.xticks(x) #设置x的刻度
    # plt.xticks(x[::2])# 当刻度太密集时候使用列表的步长(间隔取值)来解决

    # 保存图片，还可以保存为svg矢量图格式，放大不会有锯齿
    plt.savefig("./t1.png")
    plt.show() # 在执行程序的时候展示图形

def demo_temperature():
    """
    如果列表a表示10点到12点的每一分钟的气温,如何绘制折线图
    观察每分钟气温的变化情况?
    :return:
    """


    pass

if __name__=="__main__":
    # demo1()
    demo2()
    pass