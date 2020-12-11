

def linear_regression_boston():
    """
      线性回归
      波士顿房价预测。包含13个特征，506条记录。
      处理流程：
          1.加载数据
          2.数据分割
          3.数据标准化
          4.LinearRegression回归模型估计器预估房价
      :return:
      """
    # 导包
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from sklearn.externals import joblib
    from sklearn.metrics import mean_squared_error

    # 加载数据
    data_boston = load_boston()
    #分割数据
    # x_train训练集特征值，x_test测试集特征值，y_train训练集目标值，y_test测试集目标值
    x_train,x_test,y_train,y_test = train_test_split(data_boston["data"],data_boston["target"],test_size=0.25)
    # 特征工程-标准化处理。特征值和目标值都需要标准化
    # 特征值标准化
    standard_x = StandardScaler()
    x_train = standard_x.fit_transform(x_train)
    x_test = standard_x.transform(x_test)

    # 目标值标准化
    standard_y = StandardScaler()
    y_train = standard_y.fit_transform(y_train.reshape(-1,1))
    y_test = standard_y.transform(y_test.reshape(-1,1))

    #实例化估计器，估计房价
    lr_model = LinearRegression()
    #训练模型
    lr_model.fit(x_train,y_train)
    print("特征系数为：\n",lr_model.coef_)

    #房价预测
    print("预测的房价为：\n",standard_y.inverse_transform(lr_model.predict(x_test)))
    print("预测的分数为：\n",lr_model.score(x_test,y_test))
    print("均方误差为：\n", mean_squared_error(standard_y.inverse_transform(y_test),
                                         standard_y.inverse_transform(lr_model.predict(x_test))))

    #预测值和真实值的折线图
    rcParams["font.sans-serif"] = "SimHei"
    fig = plt.figure(figsize=(20,8))

    y_pred = lr_model.predict(x_test)
    plt.plot(range(y_test.shape[0]),y_test,color="blue",linewidth=1.5,linestyle="-")
    plt.plot(range(y_test.shape[0]),y_pred,color="red",linewidth=1.5)
    plt.legend(["真实值","预测值"])
    plt.show()

    # # 模型保存
    # joblib.dump(lr_model,"./re.pkl")
    #
    # # 测试保存的模型
    # model = joblib.load("./re.pkl")
    # predict_result = standard_y.inverse_transform(model.predict(x_test))
    # print(predict_result)


def SGDRegressor_boston():
    """
    线性回归之梯度下降
    波士顿房价预测。包含13个特征，506条记录。
      处理流程：
          1.加载数据
          2.数据分割
          3.数据标准化
          4.LinearRegression回归模型估计器预估房价
    :return:
    """
    #导包
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from sklearn.externals import joblib
    from sklearn.metrics import mean_squared_error

    #加载数据
    data_boston = load_boston()
    #分割数据,训练集和测试机
    x_train,x_test,y_train,y_test = train_test_split(data_boston["data"],data_boston["target"],test_size=0.25)

    #特征工程-标准化处理。
    standard_x = StandardScaler()
    x_train = standard_x.fit_transform(x_train)
    x_test = standard_x.transform(x_test)

    standard_y = StandardScaler()
    y_train = standard_y.fit_transform(y_train.reshape(-1,1))
    y_test = standard_y.transform(y_test.reshape(-1,1))

    #实例化估计器，估计房价
    sgd_model = SGDRegressor()
    #训练模型
    sgd_model.fit(x_train,y_train)
    print("特征系数为：\n",sgd_model.coef_)

    #预估房价
    print("预估的房价为：\n",standard_y.inverse_transform(sgd_model.predict(x_test))[:2])
    print("预测的分数为：\n",sgd_model.score(x_test,y_test))
    print("均方误差为：\n",mean_squared_error(standard_y.inverse_transform(y_test),
                                        standard_y.inverse_transform(sgd_model.predict(x_test))))

    #预测房价与真实房价的折线图
    rcParams["font.sans-serif"] = "SimHei"
    fig = plt.figure(figsize=(20,8))

    y_pred = sgd_model.predict(x_test)
    #真实值
    plt.plot(range(y_test.shape[0]),y_test,color="blue",linewidth=1.5,linestyle="-")
    #预测值
    plt.plot(range(y_test.shape[0]),y_pred,color="red",linewidth=1.5)
    plt.legend(["真实值","预测值"])
    plt.show()

    #模型保存
    # joblib.dump(sgd_model,"./sgd.pkl")

    #测试保存的模型
    # model = joblib.load("./sgd.pkl")
    # predict_result = standard_y.inverse_transform(model.predict(x_test))
    # print(predict_result)

def ridge():
    """
    岭回归
    波士顿房价预测。包含13个特征，506条记录。
    :return:
    """
    #导包
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    #加载数据
    boston = load_boston()
    #分割数据集，分割为训练集和测试机
    x_train,x_test,y_train,y_test = train_test_split(boston["data"],boston["target"],test_size=0.25)

    #标准化处理特征值和目标值
    standard_x = StandardScaler()
    x_train = standard_x.fit_transform(x_train)
    x_test= standard_x.transform(x_test)

    standard_y = StandardScaler()
    y_train = standard_y.fit_transform(y_train.reshape(-1,1))
    y_test = standard_y.transform(y_test.reshape(-1,1))

    #岭回归估计器模型训练
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_train,y_train)

    #预测房价
    print("预测的房价为：", standard_y.inverse_transform(ridge_model.predict(x_test))[:3])
    print("预测评分为：",ridge_model.score(x_test,y_test))
    print("均方误差为：",mean_squared_error(standard_y.inverse_transform(y_test),standard_y.inverse_transform(ridge_model.predict(x_test))))






if __name__=="__main__":

    """
    回归算法1：线性回归(最小二乘回归)
    概念：通过一个或者多个自变量与因变量之间进行建模的回归分析。 
    如何判断拟合函数好坏？损失函数。通过最小化每个数据点到线的垂直偏差平方和来判断损失大小，进而确定损失函数。
    API：
        sklearn.linear_model.LinearRegression 最小二乘之正规方程
            特征比较复杂时，求解速度慢。
        sklearn.linear_model.SGDRegressor  最小二乘之梯度下降
            可以处理数据规模比较大的任务
    回归评估：均方误差
        API：sklearn.metrics.mean_squared_error
            mean_squared_error(y_ture,y_perd),输入真实值与预测值
        注意：如果使用了标准化，需要转换到原始的数据进行回归评估
    """
    # linear_regression_boston()


    """
    回归算法2： 线性回归(梯度下降)
    概念：通过一个或者多个自变量与因变量之间进行建模的回归分析。 
    如何判断拟合函数好坏？损失函数。通过最小化每个数据点到线的垂直偏差平方和来判断损失大小，进而确定损失函数。
    API：
        sklearn.linear_model.LinearRegression 最小二乘之正规方程
            特征比较复杂时，求解速度慢。
        sklearn.linear_model.SGDRegressor  最小二乘之梯度下降
            可以处理数据规模比较大的任务
    回归评估：均方误差
        API：sklearn.metrics.mean_squared_error
            mean_squared_error(y_ture,y_perd),输入真实值与预测值
        注意：如果使用了标准化，需要转换到原始的数据进行回归评估
    """
    # SGDRegressor_boston()

    """
    回归算法3：岭回归
        概念：带有正则化的回归，解决线性回归过拟合问题
        API：sklearn.linear_model.Ridge 
        岭回归回归得到的回归系数更符合实际，更可靠。另外，能让估计参数的波动范围变小，
        变的更稳定。在存在病态数据偏多的研究中有较大的实用价值。
    """
    ridge()