def knn_breast_cancer():
    """
    K-近邻算法
    威斯康星州乳腺癌数据集
    含有30个特征、569条记录、目标值为0或1
    :return:
    """
    # 导包
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV
    # 导入数据
    data_cancer = load_breast_cancer()
    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        data_cancer.data, data_cancer.target, test_size=0.25)
    # 数据标准化处理
    stdScaler = StandardScaler().fit(x_train)
    x_trainStd = stdScaler.transform(x_train)
    x_testStd = stdScaler.transform(x_test)
    # 使用KNeighborsClassifier函数构建knn模型
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_trainStd, y_train)
    param_grid = {"n_neighbors": [1, 3, 5, 7]}
    grid_search = GridSearchCV(knn_model, param_grid=param_grid, cv=5).fit(x_trainStd, y_train)
    print("网格搜索中最佳结果的参数设置：", grid_search.best_params_)
    print("网格搜索中最高分数估计器的分数为：", grid_search.best_score_)
    print("测试集准确率为：", knn_model.score(x_testStd, y_test))
    print("每个类别的精确率和召回率：\n", classification_report(y_test, knn_model.predict(x_testStd),
                                                   target_names=data_cancer.target_names))
    print("预测测试集前5个结果为：", knn_model.predict(x_testStd)[:5])
    # print("测试集前5个最近邻点为：\n", knn_model.kneighbors(x_testStd)[0][:5],
    #       "\n测试集前5个最近邻点的距离：\n", knn_model.kneighbors(x_testStd)[1][:5])


def knn_iris():
    """
    K-近邻算法
     鸢尾花数据集
    含有3个特征、150条记录、目标值为0、1、2
    :return:
    """
    # 导包
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    # 导入数据
    data_iris = load_iris()
    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        data_iris.data, data_iris.target, test_size=0.25)
    # 数据标准化处理
    stdScaler = StandardScaler().fit(x_train)
    x_trainStd = stdScaler.transform(x_train)
    x_testStd = stdScaler.transform(x_test)
    # 使用KNeighborsClassifier函数构建knn模型
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_trainStd, y_train)
    print("测试集准确率为：", knn_model.score(x_testStd, y_test))
    print("每个类别的精确率和召回率：\n",
          classification_report(y_test, knn_model.predict(x_test), target_names=data_iris.target_names))
    print("预测测试集前5个结果为：", knn_model.predict(x_testStd)[:5])
    # print("测试集前5个最近邻点为：\n", knn_model.kneighbors(x_testStd)[0][:5],
    #       "\n测试集前5个最近邻点的距离：\n", knn_model.kneighbors(x_testStd)[1][:5])


def knn():
    """
    k-近邻算法预测用户签到位置
    :return:
    """
    import pandas as pd
    # 读取数据
    data = pd.read_csv("./")

    # 处理数据
    # 1.缩小数据范围，查询数据信息
    data = data.query("x>1.0 & x<1.25 & y>2.5 & y<2.75")
    # 处理时间的数据
    time_value = pd.to_datetime(data["time"], unit="s")
    # 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造特征值
    data["day"] = time_value.day
    data["hour"] = time_value.hour
    data["weekday"] = time_value.weekday

    # 删除时间戳特征
    data = data.drop(["time"], axis=1)

    # 把签到的数量少于n个目标位置删除
    place_count = data.groupby("place_id").count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data["place_id"].isin(tf.place_id)]

    # 取出数据中的特征值和目标值
    y = data["place_id"]
    x = data.drop(["place_id"], axis=1)


def naive_bayes():
    """
    朴素贝叶斯进行文本分类
    对20类新闻数据进行预估
    处理流程：
        1.加载20类新闻数据集，并进行分割
        2.生成文章特征词
        3.朴素贝叶斯估计器预估
    :return:
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    # 加载数据
    news = fetch_20newsgroups(subset="all")
    # 数据分割
    print(news.data[:2])
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 对数据进行特征抽取,实例化TF-IDF
    tf = TfidfVectorizer()
    # 以训练集中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    # print(tf.get_feature_names())
    x_test = tf.transform(x_test)
    print(x_train.shape)
    print(x_test.shape)
    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测测试集前10个结果：", y_predict[:10])
    # 获取预测的准确率
    print("测试集准确率为：", mlt.score(x_test, y_test))
    return None


def decision_tree_iris():
    """
    决策树鸢尾花数据集
    :return:
    """
    # 导包
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    import graphviz
    # 加载数据
    data_iris = load_iris()
    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(data_iris.data, data_iris.target, test_size=0.25)
    # 实例化决策树估计器
    decision_tree_model = DecisionTreeClassifier()
    # 训练数据
    decision_tree_model.fit(x_train, y_train)
    # 测试集得分
    print("测试集得分: ", decision_tree_model.score(x_test, y_test))
    # 决策过程可视化，安装graphviz,windows运行需要添加到环境变量，结果输出为pdf文件
    dot_data = tree.export_graphviz(decision_tree_model,out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    pass


def decision_tree():
    """
    决策树：
    泰坦尼克号乘客生存分类分析
    处理流程：
        1.pd读取数据
        2.选择有影响的特征
        3.处理缺失值
        4.进行特征工程，pd转换字典，特征抽取
        5.决策树估计器预估
    :return:
    """
    # 导包
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    # 加载数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据，找出特征值和目标值
    x = titan[["pclass", "age", "sex"]]
    print(type(x))
    y = titan["survived"]
    # print(x["age"])
    # 缺失值处理。
    # inplace=True,不创建新的对象，直接对原始对象进行修改
    # inplace=False,对数据进行修改，创建并返回新的对象接收修改的结果
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据集为训练集和测试机
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程：DictVectorizer对非数字化数据进行特征值化 (one_hot编码)
    dict = DictVectorizer(sparse=False)
    # 调用fit_transform()输入数据并转换，输入的数据是字典格式
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    # orient="records" 形成[{column -> value}, … , {column -> value}]的结构
    # 整体构成一个列表，内层是将原始数据的每行提取出来形成字典
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # print(x_train)
    # 使用决策树估计器进行预测
    deci_tree = DecisionTreeClassifier()
    # 训练数据
    deci_tree.fit(x_train, y_train)
    print("预测的准确率：", deci_tree.score(x_test, y_test))
    # dot_data = export_graphviz(deci_tree,out_file=None,feature_names=["年龄","pclass=1st","pclass=2st","pclass=3st",'女性', '男性'])


def random_forest():
    """
    随机森林
    泰坦尼克号乘客生存分析
    :return:
    """
    # 导包
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    # 加载数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据,找出特征值和目标值
    x = titan[["pclass", "age", "sex"]]
    y = titan["survived"]
    # print(x.loc[:,"age"])

    # 缺失值处理。inplace=True不创建对象直接对原对象进行修改
    x.loc[:, "age"].fillna(x.loc[:, "age"].mean, inplace=True)
    # x["age"].fillna(x["age"].mean,inplace=True)

    # 分割数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 数据处理：对数据进行特征值化(特征抽取)。
    dict = DictVectorizer(sparse=False)

    # 训练转换数据
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))

    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 随机森林进行预测
    rf = RandomForestClassifier()
    # 超参数
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("准确率：", gc.score(x_test, y_test))
    print("查看选择的参数：", gc.best_params_)
    return None

def random_forest_cancer():
    """
    随机森林
    威斯康星州乳腺癌数据集
    含有30个特征、569条记录、目标值为0或1
    :return:
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    # 导入数据
    data_cancer = load_breast_cancer()
    # 将数据集划分为训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(
        data_cancer.data,data_cancer.target,test_size=0.25)
    # # 数据标准化处理
    # stdScaler = StandardScaler().fit(x_train)
    # x_trainStd = stdScaler.transform(x_train)
    # x_testStd = stdScaler.transform(x_test)
    rf_model = RandomForestClassifier(n_estimators=10)
    rf_model.fit(x_train,y_train)
    print("训练出的前2个决策树的模型为：",rf_model.estimators_[0:2])
    print("预测测试集前10个结果为：",rf_model.predict(x_test)[:10])
    print("测试集准确率为:",rf_model.score(x_test,y_test))
    pass

def logistic_regression():
    """
    逻辑回归分类算法
        应用场景：
            判断用户性别，预测用户是否购买给定的商品，判断一条评论是正面还是负面
        数据：load_breast_cancer(威斯康星州乳腺癌数据)，包含30个特征，569条记录，目标值为0或1
    :return:
    """
    # 导包
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # 加载数据
    data_cancer = load_breast_cancer()
    # 将数据集划分为训练集和测试机
    x_train, x_test, y_train, y_test = train_test_split(data_cancer["data"], data_cancer["target"], test_size=0.25)

    # 数据标准化
    standard = StandardScaler()
    standard.fit(x_train)
    x_trainStd = standard.transform(x_train)
    x_testStd = standard.transform(x_test)

    # 构建模型
    logestic_model = LogisticRegression(solver="saga")
    logestic_model.fit(x_trainStd, y_train)
    print("训练的模型为：", logestic_model)
    print("模型各特征的相关系数：", logestic_model.coef_)

    # 预测测试机
    print("预测的测试机结果为：\n", logestic_model.predict(x_testStd))
    print("预测的准确率为：\n", logestic_model.score(x_testStd, y_test))
    return None





if __name__ == "__main__":
    """
    分类算法1：k近邻算法(KNN)
    概念：如果一个样本中在特定空间中的K个最相似(即特征空间中最相邻)的样本中的大多数属于某一个类别,
          则该样本也属于这个类别
    K近邻算法API:sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
        n_neighbors:邻居数，默认是5
        algorithm:选用计算最近邻居的算法，默认'auto',自动选择 
    距离采用：欧氏距离
    """
    """
    分类算法2：朴素贝叶斯(所有特征之间是条件独立的)
    API:sklearn.naive_bayes.MultinomialNB(alpha=1.0)
        alpha:拉普拉斯平滑系数，默认1
    """
    # naive_bayes()
    knn_breast_cancer()
    # knn_iris()kn

    """
    分类算法3：决策树。思想很简单就是if else
    信息熵代表信息量的大小，值越大代表信息量越大
    信息增益：得知一个信息之后，信息熵减少的大小
    API：sklearn.tree.DecisionTreeClassifier(criterion='gini',max_depth=None,random_state=None)
        criterion：默认是'gini'系数，也可以选择信息增益的熵'entropy'
        max_depth:树的深度大小
        random_state:随机数种子
    返回值：决策树的路径
    """
    # decision_tree()
    # decision_tree_iris()


    """
    分类算法4：随机森林(又叫集成学习)
        随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，本质是机器学习分支--集成学习(Ensemble learing)方法。
        每棵树都是一个分类器，那么对于一个输入样本，N棵树会有N个分类结果。随机森林集成了所有的分类投票结果，将投票次数最多的类别指定为最终的输出。
    注意：
        1.随机抽样训练集。避免训练出的树分类结果都一样
        2.有放回的抽样，避免每棵树的训练样本完全不同，没有交集，这样每棵树都是"有偏的"，都是绝对"片面的"  
    API:sklearn.ensemble.RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=None,bootstrap=True,random_state=None)
        n_estimators:森林里的树木数量，默认10
        criteria:分割特征的测量方法，默认gini系数
        max_depth:树的最大深度，默认无
        bootstrap:构建有放回的抽样，默认True
    优点：
        1.当前几个算法中具有极高的准确率 
        2.有效的运行在大数据集上
        3.能够处理高维数据，且不用降维
        4.对缺省的数据能够获得很好的结果
    """
    # random_forest()
    # random_forest_cancer()

    """
    分类算法5：逻辑回归
        逻辑回归是解决二分类问题的利器。
        sigmoid函数：输出的值在[0,1]之间，默认概率0.5阈值，超过0.5认为有可能发生，小于0.5认为不可能发生
        API：sklearn.linear_model.LogisticRegression
        
    """
    # logistic_regression()
