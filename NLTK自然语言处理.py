
import nltk
"""
NTLK是著名的Python自然语言处理工具包，但是主要针对的是英文处理。
NLTK配套有文档，有语料库，有书籍。
    1.NLP领域中最常用的一个Python库
    2.开源项目
    3.自带分类、分词等功能
    4.强大的社区支持
    5.语料库，语言的实际使用中真是出现过的语言材料
    6.http://www.nltk.org/py-modindex.html
在NLTK的主页详细介绍了如何在Mac、Linux和Windows下安装NLTK：http://nltk.org/install.html ，
建议直接下载Anaconda，省去了大部分包的安装，安装NLTK完毕，可以import nltk测试一下，如果没有问题，还有下载NLTK官方提供的相关语料。
"""
"""
语料库就是把平常我们说话的时候袭的句子、一些文学bai作品的语句段落、
报刊杂志上出现过的语句段落等等在现实生活中真实出现过的语言材料
整理在一起，形du成一个语料库，以便做科学研究的时候能够从中取材或者得到数据佐证。
"""
def ss_nltk():
    # 1.查看语料库
    #引用布朗大学的语料库
    from nltk.corpus import brown # 需要下载brown语料库
    # 查看语料库包含的类别
    print(brown.categories())
    # 查看brown语料库
    print('共有{}个句子'.format(len(brown.sents())))
    print('共有{}个单词'.format(len(brown.words())))

    # 2.NLTK词条化(分词 (tokenize))
    """
    将句子拆分成具有语言语义学上意义的词
    中、英文分词区别：
        英文中，单词之间是以空格作为自然分界符的
        中文中没有一个形式上的分界符，分词比英文复杂的多
    中文分词工具，如：结巴分词 pip install jieba
    得到分词结果后，中英文的后续处理没有太大区别
    """
    """
    nltk.sent_tokenize(text) #按句子分割 
    nltk.word_tokenize(sentence) #分词 
    nltk的分词是句子级别的，所以对于一篇文档首先要将文章按句子进行分割，然后句子进行分词
    """
    text = "Are you curious about tokenization? Let's see how it works! " \
           "We need to analyze a couple of sentences with punctuations to see it in action."
    from nltk.tokenize import sent_tokenize
    sent_tokenize_list = sent_tokenize(text)
    print(sent_tokenize_list)


    """
    词形问题
    look, looked, looking
    影响语料学习的准确度
    词形归一化
    """
    #3.词干提取(stemming)
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer

    stemmer_porter = PorterStemmer()
    stemmer_lancaster = LancasterStemmer()
    stemmer_snowball = SnowballStemmer('english')

    print(stemmer_porter.stem('looked'))
    print(stemmer_porter.stem('looking'))
    print(stemmer_porter.stem("words"))

    # 4.NLTK词形归并(lemmatization)
    """
    stemming，词干提取，如将ing, ed去掉，只保留单词主干
    lemmatization，词形归并，将单词的各种词形归并成一种形式，
    如am, is, are -> be, went->go
    NLTK中的stemmer --> PorterStemmer, SnowballStemmer, LancasterStemmer
    NLTK中的lemma --> WordNetLemmatizer
    指明词性可以更准确地进行lemma
    """
    from nltk.stem import WordNetLemmatizer # 需要下载wordnet语料库

    wordnet_lematizer = WordNetLemmatizer()
    print(wordnet_lematizer.lemmatize('cats'))
    print(wordnet_lematizer.lemmatize('boxes'))
    print(wordnet_lematizer.lemmatize('are'))
    print(wordnet_lematizer.lemmatize('went'))

    # 指明词性可以更准确地进行lemma
    # lemmatize 默认为名词
    print(wordnet_lematizer.lemmatize('are', pos='v'))
    print(wordnet_lematizer.lemmatize('went', pos='v'))

    # 5.词性标注 (Part-Of-Speech) nltk.word_tokenize()
    import nltk
    words = nltk.word_tokenize('Python is a widely used programming language.')
    print(nltk.pos_tag(words))  # 需要下载 averaged_perceptron_tagger

    # 6.NLTK的Stopwords,去除停用词
    """
    去除停用词
    为节省存储空间和提高搜索效率，NLP中会自动过滤掉某些字或词
    停用词都是人工输入、非自动化生成的，形成停用词表
    
    使用NLTK去除停用词
        stopwords.words()
    分类
        语言中的功能词，如the, is…
        词汇词，通常是使用广泛的词，如want
    中文停用词表
        中文停用词库
        哈工大停用词表
        四川大学机器智能实验室停用词库
        百度停用词列表
    其他语言停用词表
        http://www.ranks.nl/stopwords
    """
    from nltk.corpus import stopwords  # 需要下载stopwords
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    print('原始词：', words)
    print('去除停用词后：', filtered_words)

# 典型的文本预处理流程
def ss_nltk2():
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    # 原始文本
    raw_text = 'Life is like a box of chocolates. You never know what you\'re gonna get.'
    # 分词
    raw_words = nltk.word_tokenize(raw_text)
    # 词形归一化
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]
    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    print('原始文本：', raw_text)
    print('预处理结果：', filtered_words)

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
    res = vector.fit_transform(["life is short,i like python",
                                "life is too long,i dislike python"])
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
    # data = vector.fit_transform(
    # ["人生苦短，我用python","人生漫长，不用python"])
    data = vector.fit_transform(
        ["人生苦短，我喜欢python","人生漫长，不用python"])
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
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，"
                     "但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，"
                     "这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。"
                     "了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

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
    a = [c1,c2,c3]
    print(a)
    data = cv.fit_transform([c1,c2,c3])
    # 获取数据特征值
    print(cv.get_feature_names())
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

if __name__=="__main__":
    # ss_nltk()
    # ss_nltk2
    countvec()
    # countvec_chinese()
    # countvec_chinese_jieba()
    tfidf_countvec_chinese_jieba()
    pass
