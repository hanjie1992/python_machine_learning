
def dictvec():
    """
    字典数据抽取：对字典数据进行特征值化
    :return:None
    """
    #实例化DictVectorizer
    # sklearn.feature_extraction特征抽取API
    from sklearn.feature_extraction import DictVectorizer
    dict = DictVectorizer(sparse=False)
    #调用fit_transform方法输入数据并转换，返回矩阵形式数据
    data = dict.fit_transform([{'city': '北京','temperature': 100},{'city': '上海','temperature':60},
                               {'city': '深圳','temperature': 30}])
    #获取特征值
    print(dict.get_feature_names())
    # 获取转换之前数据
    print(dict.inverse_transform(data))
    #转换后的数据
    print(data)
    return None

def countvec():
    """
    英文文本特征抽取：对文本数据进行特征值化
    :return:
    """
    #导入包
    from sklearn.feature_extraction.text import CountVectorizer
    #实例化CountVectorizer()
    vector = CountVectorizer()
    #调用fit_transform输入并转换数据
    res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
    # 获取特征值
    print(vector.get_feature_names())
    # 转换后的数据
    print(res.toarray())
    return None

def countvec_chinese():
    """
    中文文本特征抽取：
    存在问题：对中文分词有误
    解决办法：使用jieba分词
    :return:
    """
    from sklearn.feature_extraction.text import CountVectorizer
    #实例化CountVectorizer()
    vector = CountVectorizer()
    #调用fit_transfrom输入并转换数据
    # data = vector.fit_transform(["人生苦短，我用python","人生漫长，不用python"])
    data = vector.fit_transform(["人生 苦短，我 喜欢 python","人生 漫长，不用 python"])
    #获取数据特征值
    print(vector.get_feature_names())
    #转换后的数据
    print(data.toarray())
    return None

def jieba_cutword():
    """
    利用jieba.cut进行分词,返回词语生成器。
    将分词结果变成字符串当作fit_transform的输入值
    :return:
    """
    import jieba
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    #转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    print(content1)
    #列表转字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1,c2,c3

def countvec_chinese_jieba():
    """
    使用jieba分词对中文进行分词
    :return:
    """
    from sklearn.feature_extraction.text import CountVectorizer
    c1,c2,c3 = jieba_cutword()
    # print(c1,c2,c3)
    #实例化CountVectorizer()
    cv = CountVectorizer()
    # 调用fit_transfrom输入并转换数据
    data = cv.fit_transform([c1,c2,c3])
    # 获取数据特征值
    # print(cv.get_feature_names())
    # 转换后的数据
    print(data.toarray())
    return None

def tfidf_countvec_chinese_jieba():
    """
    TF-IDF-文本词语占比分析:
    TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，
    则认为此词或者短语具有很好的类别区分能力，适合用来分类。
    :return:
    """
    #导包
    from sklearn.feature_extraction.text import TfidfVectorizer
    #字符串
    c1,c2,c3 = jieba_cutword()
    #实例化TF-IDF
    tf = TfidfVectorizer()
    #调用fit_transform()输入并转换数据
    data = tf.fit_transform([c1,c2,c3])
    #获取数据特征值
    print(tf.get_feature_names())
    #转换后的数据
    print(data.toarray())
    return None

def min_max_scaler():
    """
    归一化处理：
    通过对原始数据进行变换把数据映射到(默认为[0,1])之间
    缺点:最大值与最小值非常容易受异常点影响，这种方法鲁棒性较差，只适合传统精确小数据场景。
    :return:
    """
    from sklearn.preprocessing import MinMaxScaler
    #实例化MinMaxScaler()
    mm = MinMaxScaler()
    # 调用fit_transform()输入并转换数据
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    # 转换后的数据
    print(data)
    return None

def standard_scaler():
    """
    标准化方法：通过对原始数据进行变换把数据变换到均值为0，方差为1范围内
    如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大，从而方差改变较小。
    :return:
    """
    from sklearn.preprocessing import StandardScaler
    #实例化StandardScaler()
    standard = StandardScaler()
    #调用fit_transform()输入并转换数据
    data = standard.fit_transform([[1,-1,3],[2,4,2],[4,6,1]])
    #转换后的数据
    print(data)
    return None

def imputer():
    """
    缺失值处理
    :return:
    """
    from sklearn.impute import SimpleImputer
    import numpy as np
    #实例化SimpleImputer(),指定缺失值为missing_values=np.NaN,
    # 填补策略为平均值strategy="mean"
    im = SimpleImputer(missing_values=np.NaN,strategy="mean")
    # 调用fit_transform()输入并转换数据
    data = im.fit_transform([[1,2],[np.NaN,3],[7,6]])
    # 转换后的数据
    print(data)
    return None

def variance_threshold():
    """
    特征选择-过滤式：删除低方差特征
    :return:
    """
    from sklearn.feature_selection import VarianceThreshold
    #实例化VarianceThreshold()
    var = VarianceThreshold()
    #调用fit_transform()输入并转换数据
    data = var.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])
    #转换后的数据
    print(data)
    return None

def pca():
    """
    PCA主成分分析进行特征降维
    :return:
    """
    #导包
    from sklearn.decomposition import PCA
    #实例化PCA,
    # n_components小数是百分比：n_components=0.9保留90%数据，一般取值90%-95%
    #n_components整数是保留的特征数量，一般不用
    pca = PCA(n_components=0.9)
    #输入并转换数据
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    #转换后的数据
    print(data)
    return None


if __name__=="__main__":
    """
    特征工程是将原始数据转换为更好地代表预测模型的潜在问题的特征的过程，从而提高了对未知数据的模型准确性。
    意义：直接影响模型的预测结果
    """
    """
    数据预处理步骤1：特征抽取
    特点：特征抽取针对非连续型数据，特征抽取对文本等进行特征值化。
          特征值化是为了计算机更好的去理解数据
    分类：字典特征抽取和文本特征抽取
    """
    dictvec()
    # countvec()
    # countvec_chinese()
    # jieba_cutword()
    # countvec_chinese_jieba()

    """ 
    数据预处理步骤2：特征处理
    通过特定的统计方法（数学方法）将数据转换成算法要求的数据
    方法：
        数值型数据：标准缩放：	1、归一化 2、标准化
        类别型数据：one-hot编码
        时间类型：时间的切分
    """
    # min_max_scaler()
    # standard_scaler()

    """
    数据预处理步骤3:缺失值处理
    方法：
        1.删除：如果每列或者行数据缺失值达到一定的比例，建议放弃整行或者整列
        2.插补：可以通过缺失值每行或者每列的平均值、中位数来填充
    API：Imputer包
    """
    # imputer()

    """
    数据预处理步骤4：特征选择
    概念：特征选择就是单纯地从提取到的所有特征中选择部分特征作为训练集特征，
         特征在选择前和选择后可以改变值、也不改变值，但是选择后的特征维数肯定比选择前小，
         毕竟我们只选择了其中的一部分特征。
    方法：
        1.过滤：variance threshold 删除所有低方差特征,默认值是保留所有非零方差特征，
               即删除所有样本中具有相同值的特征。
        
    """
    # variance_threshold()
    """
    数据预处理步骤5：降维(PCA)
    PCA，是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
    可以削减回归分析或者聚类分析中特征的数量
    """
    # pca()