

def k_means():
    """
    k-means聚类
    数据集：iris鸢尾花数据集
    :return:
    """
    #导包
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    #加载数据
    data_iris = load_iris()
    #构建并训练K-Means模型
    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(data_iris["data"])
    print("训练的模型为：",kmeans_model)

    #画图展示
    #绘制iris原本的数据类别
    plt.scatter(data_iris.data[:,0],data_iris.data[:,1],c=data_iris.target)
    plt.show()
    #绘制kmeans聚类结果
    y_predict = kmeans_model.predict(data_iris.data)
    plt.scatter(data_iris.data[:,0],data_iris.data[:,1],c=y_predict)
    plt.show()
    return None

if __name__ =="__main__":
    """
    聚类算法：
        是在没有给定划分类别的情况下，根据数据相似度进行样本分组的一种方法。
        聚类的输入是一组未被标记的样本，聚类根据数据自身的距离或相似度将他们划分为若干组，
        划分的原则是组内样本最小化组间距离最大化
        
    """
    """
    聚类算法1：K-means:
        步骤：
            1.随机设置k个特征空间内的点作为初始的聚类中心
            2.对于其他每个点计算到K个中心得距离，未知类的点选择距离最近的一个聚类中心点作为自己的属于的类
            3.接着对标记的聚类中心之后，重新计算出每个聚类的新中心点
            4.如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程
    """
    k_means()