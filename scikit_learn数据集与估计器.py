

def iris():
    """
    加载鸢尾花数据集(分类数据集)
    :return:
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    print("鸢尾花数据集的特征值为：\n",iris.data)
    print("鸢尾花数据集的目标值为：\n",iris.target)
    print("鸢尾花数据集概览信息：\n",iris.DESCR)

    #鸢尾花数据集划分为训练集和测试集
    #注意返回值顺序，固定的。
    #x_train训练集特征值，y_train训练集目标值；x_test测试集特征值，y_test测试集目标值
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25)
    print("训练集特征值和目标值：",x_train,y_train)
    print("测试集特征值和目标值：",x_test,y_test)

def news_groups():
    """
    20类新闻文章数据集(分类数据集)
    :return:
    """
    #导入包
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    #加载数据，subset="all"：选择所有数据（包括训练集和测试机）
    news = fetch_20newsgroups(subset="all")
    print("获取特征值：",news.data)
    print("获取目标值：",news.target)
    #新闻文章数据集划分为训练集和测试机
    #注意数据返回顺序，固定
    #x_train训练集特征值，y_train训练集目标值；x_test测试机特征值，y_test测试机目标值
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)
    print("训练集特征值和目标值：",x_train,y_train)
    print("测试机特征值和目标值：",x_test,y_test)

def boston():
    """
    波士顿房价数据集(回归数据集)
    :return:
    """
    #导包
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    print("波士顿房价数据特征值：",boston.data)
    print("波士顿房价数据目标值：",boston.target)
    # x_train训练集特征值，y_train训练集目标值；x_test测试机特征值，y_test测试机目标值
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.25)
    print("波士顿房价训练集特征值和目标值：",x_train,y_train)
    print("波士顿房价测试机特征值和目标值：",x_test,y_test)



if __name__=="__main__":

    """
    sklearn数据集
        问题：自己准备数据集，耗时耗力，不一定真实
        (1)数据集划分：
            1.训练数据：用于训练，构建模型
            2.测试数据：在模型检验时使用，用于评估模型是否有效
        (2)数据集API：sklearn.datasets
            1.datasets.load_*() 获取小规模数据集，数据保护在datasets里
            2.datasets.fetch_*(data_home=None) 获取大规模数据集，需要从网络上下载。
              函数的第一个参数是的目录。
            返回值：
                load*和fetch*返回的数据类型是datasets.base.Bunch(字典格式)
                    data:特征数据数组，是[n_sample*n_features]几行几列的二维numpy.ndarray数组
                    target:标签数组，是n_samples的一维numpy.ndarray数组
                    feature_names:特征名字（新闻数据，手写数字，回归数据集没有）
                    target_names:标签名，回归数据集没有
         (3)数据集分割API：sklearn.model_selection.train_test_split()
            输入：数据集特征值、数据集目标值、测试集比例大小
            返回值：训练集特征值、测试集特征值、训练集目标值、测试集特征值
    """
    # iris()
    # news_groups()
    # boston()

    """
    sklearn估计器(estimator)：是一类实现了算法的API。分类器和回归器都属于estimator
    估计器常见API：
        1.sklearn.neighbors k-近邻
        2.sklearn.naive_bayes 朴素贝叶斯
        3.sklearn.linear_model.LogisticRegression 逻辑回归
        4.sklearn.linear_model.LinearRegression 线性回归
        5.sklearn.linear_model.Ridge 岭回归
    """