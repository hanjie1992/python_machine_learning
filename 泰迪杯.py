import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
def taidi1():
    """
    泰迪杯第一问
    :return:
    """
    # 获取目标值数据、特征值数据文件
    df2 = pd.read_excel("../data/2.xlsx")
    # 去除空格，制表符
    df2["留言详情"] = df2["留言详情"].apply(lambda x:x.replace('\n', '').replace('\t', ''))
    # 2个series合并为DataFrame
    # df_data = list(zip(df2["留言详情"],df2["留言主题"]))
    df_data = np.array(df2["留言详情"])
    df_target = np.array(df2["一级分类"])
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25)
    # 对数据进行特征抽取,实例化TF-IDF
    tf = TfidfVectorizer()
    # 以训练集中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    f1 = model_selection.cross_val_score(mlt,
                                         x_train, y_train, scoring='f1_weighted', cv=5)
    print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
    print("预测测试集前10个结果：", y_predict[:10])
    # 获取预测的准确率
    print("测试集准确率为：", mlt.score(x_test, y_test))
    return None


def taidi():
    """
    泰迪杯第一问
    :return:
    """
    # 获取目标值数据、特征值数据文件
    df2 = pd.read_excel("../data/2test.xlsx")
    # 去除空格，制表符
    df2["留言详情"] = df2["留言详情"].apply(lambda x:x.replace('\n', '').replace('\t', ''))
    # 2个series合并为DataFrame
    # df_data = list(zip(df2["留言详情"],df2["留言主题"]))
    df_data = df2["留言详情"].values.tolist()
    # df_target = df2["一级标签"].values.tolist()
    # # 数据分割
    # x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25)
    # print(type(x_test))
    # print(type(df_data))
    # # 对数据进行特征抽取,实例化TF-IDF
    # tf = TfidfVectorizer()
    # # 以训练集中的词的列表进行每篇文章重要性统计
    # x_train = tf.fit_transform(x_train)
    # x_test = tf.transform(x_test)
    # print(x_train.shape)
    # print(x_test.shape)
    # # 进行朴素贝叶斯算法的预测
    # mlt = MultinomialNB(alpha=1.0)
    # mlt.fit(x_train, y_train)
    # y_predict = mlt.predict(x_test)
    # f1 = model_selection.cross_val_score(mlt,
    #                                      x_train, y_train, scoring='f1_weighted', cv=5)
    # print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")
    # print("预测测试集前10个结果：", y_predict[:10])
    # # 获取预测的准确率
    # print("测试集准确率为：", mlt.score(x_test, y_test))
    #
    from sklearn.externals import joblib
    # joblib.dump(mlt, "./mlt.pkl")
    #
    # # 测试保存的模型
    model = joblib.load("./mlt.pkl")
    # predict_result = standard_y.inverse_transform(model.predict(x_test))
    predict_result = model.predict([df_data])
    print(predict_result)


    return None


def taidi3():
    """
    泰迪杯第一问
    :return:
    """
    # 获取目标值数据、特征值数据文件
    df2 = pd.read_excel("../data/2.xlsx")
    # 去除空格，制表符
    df2["留言详情"] = df2["留言详情"].apply(lambda x:x.replace('\n', '').replace('\t', ''))
    # 2个series合并为DataFrame
    # df_data = list(zip(df2["留言详情"],df2["留言主题"]))
    df_data = df2["留言详情"].values.tolist()
    sent_words = [list(jieba.cut(sent)) for sent in df_data]
    document = [" ".join(sent) for sent in sent_words]
    # 对数据进行特征抽取,实例化TF-IDF
    tfidf_model = TfidfVectorizer().fit(document)
    # print(tfidf_model.vocabulary_)
    sparse_result = tfidf_model.transform(document)
    print(sparse_result)
    return None


def taidi4():
    df2 = pd.read_excel("../data/2.xlsx")
    # 去除空格，制表符 
    df2["留言详情"] = df2["留言详情"].apply(lambda x: x.replace('\n', '').replace('\t', ''))
    df_data = df2["留言详情"].values.tolist()
    df_target = df2["一级标签"].values.tolist()
    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.25)
    # 实例化CountVectorizer()
    vectorizer = CountVectorizer()
    # 调用fit_transfrom输入并转换数据
    words = vectorizer.fit_transform(x_train)
    test_word = vectorizer.transform(x_test)
    # 实例化多项式分布的朴素贝叶斯
    clf = MultinomialNB().fit(words,y_train)
    predicted = clf.predict(test_word)
    for doc,category in zip(x_test,predicted):
        print(doc,":",category)
    print("每个类别的精确率和召回率：\n",classification_report(y_test, predicted))
    pass
if __name__=="__main__":
    taidi()
    # taidi1()
    # taidi3()
    # taidi4()
    pass