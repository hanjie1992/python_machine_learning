

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


def hierarchical_clustering():
    """
    层次聚类--鸢尾花数据集
    :return:
    """
    # 导包
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering
    import matplotlib.pyplot as plt
    # 加载数据
    data_iris = datasets.load_iris()
    x = data_iris.data
    y = data_iris.target
    clusting_ward = AgglomerativeClustering(n_clusters=3)
    clusting_ward.fit(x)
    print("簇类别标签为：",clusting_ward.labels_)
    print("叶子节点数量：",clusting_ward.n_leaves_)
    # 训练模型并预测每个样本的簇标记
    cw_ypre = AgglomerativeClustering(n_clusters=3).fit_predict(x)
    # 绘制散点图
    plt.scatter(x[:,0],x[:,1],c=cw_ypre)
    plt.title("ward linkage",size=18)
    plt.show()


def dbscan_clustering():
    """
    DBSCAN(密度聚类)
    :return:
    """
    # 导包
    from sklearn import datasets
    from sklearn.cluster import DBSCAN
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    # 生成两簇非凸数据
    """
    make_blobs函数是为聚类产生数据集,产生一个数据集和相应的标签
    n_samples:表示数据样本点个数,默认值100
    n_features:表示数据的维度，默认值是2
    centers:产生数据的中心点，默认值3
    cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
    center_box：中心确定之后的数据边界，默认值(-10.0, 10.0)
    shuffle ：洗乱，默认值是True
    random_state:官网解释是随机生成器的种子
    """
    x1,y2 = datasets.make_blobs(n_samples=1000,n_features=2,centers=[[1.2,1.2]],cluster_std=[[0.1]],random_state=9)
    # 一簇对比数据
    """
    datasets.make_circles()专门用来生成圆圈形状的二维样本.
    factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点，
    里圈代表一个类，外圈也代表一个类.noise表示有0.05的点是异常点
    """
    x2,y1 = datasets.make_circles(n_samples=5000,factor=0.6,noise=0.05)
    x = np.concatenate((x1,x2))
    plt.scatter(x[:,0],x[:,1],marker="o")
    plt.show()

    dbs = DBSCAN().fit(x)
    print("DBSCAN模型的簇标签：",dbs.labels_)
    print("核心样本的位置为：",dbs.core_sample_indices_)
    #通过簇标签看出，默认参数的DBSCAN模型将所有样本归为一类，与实际不符，
    #调整eps参数和min_samples参数优化聚类效果

    ds_pre = DBSCAN(eps=0.1,min_samples=12).fit_predict(x)
    plt.scatter(x[:,0],x[:,1],c=ds_pre)
    plt.title("DBSCAN",size=17)
    plt.show()

    #与k-means对比一下
    km_pre = KMeans(n_clusters=3,random_state=9).fit_predict(x)
    plt.scatter(x[:,0],x[:,1],c=km_pre)
    plt.title("K-means",size=17)
    plt.show()


def gmm_clustering():
    """
    高斯混合模型（GMM）--iris数据集案例
    :return:
    """
    # 导包
    from sklearn import datasets
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # 加载数据
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    #绘制样本数据
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title("iris",size=18)
    plt.show()

    # 构建聚类为3的GMM模型
    gmm_model = GaussianMixture(n_components=3).fit(x)
    print("GMM模型的权重为：",gmm_model.weights_)
    print("GMM模型的均值为：",gmm_model.means_)
    #获取GMM模型聚类结果
    gmm_pre = gmm_model.predict(x)
    plt.scatter(x[:,0],x[:,1],c = gmm_pre)
    plt.title("GMM",size=18)
    plt.show()
    #K-means聚类
    km_pre = KMeans(n_clusters=3).fit_predict(x)
    plt.scatter(x[:,0],x[:,1],c=km_pre)
    plt.title("K-means",size=18)
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

    """
    聚类算法2：层次聚类
    """
    # hierarchical_clustering()

    """
    DBSCAN(密度聚类)
    """
    # dbscan_clustering()

    """
    高斯混合模型（GMM）
    """
    # gmm_clustering()