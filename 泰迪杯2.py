import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def message_classification():
    """
    第一问：群众留言分类

    在处理网络问政平台的群众留言时，工作人员首先按照一定的划分体系（参考附件 1 提
    供的内容分类三级标签体系）对留言进行分类，以便后续将群众留言分派至相应的职能部门
    处理。目前，大部分电子政务系统还是依靠人工根据经验处理，存在工作量大、效率低，且
    差错率高等问题。请根据附件 2 给出的数据，建立关于留言内容的一级标签分类模型。
    通常使用 F-Score 对分类方法进行评价
    :return:
    """
    # 获取目标值数据、特征值数据文件
    data_message2 = pd.read_excel("../data/2test.xlsx")
    data_message1 = pd.read_excel("../data/2.xlsx")
    # 去除留言详情列空格，制表符
    data_message1["留言详情"] = data_message1["留言详情"].apply(lambda x: x.replace('\n', '').replace('\t', ''))
    data_message2["留言详情"] = data_message2["留言详情"].apply(lambda x: x.replace('\n', '').replace('\t', ''))
    # 特征值数据
    data = data_message1["留言详情"]
    data2 = data_message2["留言详情"]
      # 目标值数据
    target = data_message1["一级标签"].values.tolist()
    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
    # 实例化CountVectorizer()。CountVectorizer会将文本中的词语转换为词频矩阵，
    # 它通过fit_transform函数计算各个词语出现的次数
    vectorizer = CountVectorizer()
    # 调用fit_transfrom输入并转换数据
    words = vectorizer.fit_transform(x_train)
    test_word = vectorizer.transform(data2)
    # 实例化多项式分布的朴素贝叶斯
    clf_model = MultinomialNB().fit(words, y_train)
    predicted = clf_model.predict(test_word)
    df = pd.DataFrame(predicted, columns=['一级标签'])
    df.to_excel("./a.xlsx",index=False)
    print("===",predicted)
    # for doc, category in zip(x_test, predicted):
    #     print(doc, ":", category)
    # print("每个类别的精确率和召回率：\n", classification_report(y_test, predicted))
    f1 = cross_val_score(clf_model, words, y_train, scoring="f1_weighted", cv=5)
    print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
    # 模型保存
    # joblib.dump(clf_model, "./clf_model.pkl")
    #
    # # # 测试保存的模型
    # model = joblib.load("./clf_model.pkl")
    # # predict_result = standard_y.inverse_transform(model.predict(x_test))
    # predict_result = model.predict(data)
    # print(predict_result)
    pass

def hot_mining():
    """
     2、热点问题挖掘
    某一时段内群众集中反映的某一问题可称为热点问题，如“XXX 小区多位业主多次反映
    入夏以来小区楼下烧烤店深夜经营导致噪音和油烟扰民”。及时发现热点问题，有助于相关
    部门进行有针对性地处理，提升服务效率。请根据附件 3 将某一时段内反映特定地点或特定
    人群问题的留言进行归类，定义合理的热度评价指标，并给出评价结果，按表 1 的格式给出
    排名前 5 的热点问题，并保存为文件“热点问题表.xls”。按表 2 的格式给出相应热点问题
    对应的留言信息，并保存为“热点问题留言明细表.xls”。
    :return:
    """
    # 获取目标值数据、特征值数据文件
    data_message = pd.read_excel("../data/3.xlsx")
    # 特征值数据
    document = data_message["留言主题"].values.tolist()
    # sent_words = [list(jieba.cut(sent)) for sent in data]
    # document = [" ".join(sent) for sent in sent_words]
    # 对数据进行特征抽取,实例化TF-IDF
    tf = TfidfVectorizer(max_df=0.9)
    # 以训练集中的词的列表进行每篇文章重要性统计
    tfidf_model  = tf.fit(document)
    sparse_result = tfidf_model.transform(document)
    # 词语与列的对应关系
    print(tfidf_model.vocabulary_)
    # 得到tf-idf矩阵，稀疏矩阵表示法
    print(sparse_result)
    # 转化为更直观的一般矩阵
    print(sparse_result.todense())
    pass


def hot_mining2():
    """
     2、热点问题挖掘
    某一时段内群众集中反映的某一问题可称为热点问题，如“XXX 小区多位业主多次反映
    入夏以来小区楼下烧烤店深夜经营导致噪音和油烟扰民”。及时发现热点问题，有助于相关
    部门进行有针对性地处理，提升服务效率。请根据附件 3 将某一时段内反映特定地点或特定
    人群问题的留言进行归类，定义合理的热度评价指标，并给出评价结果，按表 1 的格式给出
    排名前 5 的热点问题，并保存为文件“热点问题表.xls”。按表 2 的格式给出相应热点问题
    对应的留言信息，并保存为“热点问题留言明细表.xls”。
    :return:
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    # 获取目标值数据、特征值数据文件
    data_message = pd.read_excel("../data/3.xlsx")
    # 特征值数据
    document = data_message["留言主题"].values.tolist()
    # sent_words = [list(jieba.cut(sent)) for sent in document]
    # document = []
    # for sent in sent_words:
    #     for x in sent:
    #         document.append(x)
    # document = [" ".join(sent) for sent in sent_words]
    result = pd.DataFrame(document,columns=["留言主题"])
    frequencies = result.groupby(by = ['留言主题'])["留言主题"].count()
    frequencies = frequencies.sort_values(ascending=False)
    frequencies_dataframe = frequencies.to_frame()
    # data_message = frequencies_dataframe.index.droplevel()
    a = pd.merge(data_message,frequencies_dataframe,on=["留言主题"])
    print(a)


    # backgroud_Image = plt.imread('../data/pl.jpg')
    # wordcloud = WordCloud(font_path="G:/workspace/font/STZHONGS.ttf",
    #                       max_words=20,
    #                       background_color='white',
    #                       mask=backgroud_Image)
    # my_wordcloud = wordcloud.fit_words(frequencies)
    # plt.imshow(my_wordcloud)
    # plt.axis('off')
    # plt.show()
    pass

def evaluation_scheme():
    """
    3、答复意见的评价
    针对附件 4 相关部门对留言的答复意见，从答复的相关性、完整性、可解释性等角度对
    答复意见的质量给出一套评价方案，并尝试实现。

    :return:
    """
    # 获取目标值数据、特征值数据文件
    data_message = pd.read_excel("../data/4.xlsx")
    # 特征值数据
    document = data_message["答复意见"].values.tolist()
    print(document)
    pass

if __name__=="__main__":
    message_classification()
    # hot_mining()
    # hot_mining2()
    # evaluation_scheme()
    pass