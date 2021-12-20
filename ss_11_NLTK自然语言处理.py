
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
def nltk_brown():
    # 1.查看语料库
    #引用布朗大学的语料库
    from nltk.corpus import brown # 需要下载brown语料库
    # 查看语料库包含的类别
    print(brown.categories())
    # 查看brown语料库
    print('共有{}个句子'.format(len(brown.sents())))
    print('共有{}个单词'.format(len(brown.words())))

def nltk_tokenize():
    """
     1.实验实现词条化
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
    # 1）创建一个text字符串，作为样例的文本：
    text = "Are you curious about tokenization?" \
           " Let's see how it works! We need to " \
           "analyze a couple of sentences with " \
           "punctuations to see it in action."
    # 2）加载NLTK模块：
    from nltk.tokenize import sent_tokenize
    """
     3）调用NLTK模块的sent_tokenize()方法，对text文本进行词条化，
    sent_tokenize()方法是以句子为分割单位的词条化方法：
    """
    sent_tokenize_list = sent_tokenize(text)
    # 4）输出结果：
    print("\nSentence tokenizer:")
    print(sent_tokenize_list)
    """
     5）调用NLTK模块的word_tokenize()方法，对text文本进行词条化，
    word_tokenize()方法是以单词为分割单位的词条化方法：
    """
    from nltk.tokenize import word_tokenize
    print("\nWord tokenizer:")
    print(word_tokenize(text))
    """
    6）最后一种单词的词条化方法是WordPunctTokenizer()，
     使用这种方法我们将会把标点作为保留对象。
    """
    # from nltk.tokenize import WordPunctTokenizer
    # word_punct_tokenizer = WordPunctTokenizer()
    # print("\nWord punct tokenizer:")
    # print(word_punct_tokenizer.tokenize(text))
    pass

def nltk_stemming():
    """2.实验实现词干还原"""
    # （1）导入词干还原相关的包：
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
    # （2）创建样例：
    words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog',
             'the', 'beaches', 'grounded', 'dreamt', 'envision']
    # （3）调用NLTK模块中三种不同的词干还原方法：
    stemmer_porter = PorterStemmer()
    stemmer_lancaster = LancasterStemmer()
    stemmer_snowball = SnowballStemmer('english')
    # （4）设置打印输出格式：多值参数
    stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']
    formatted_row = '{:>16}' * (len(stemmers) + 1)
    print('\n', formatted_row.format('WORD', *stemmers), '\n')
    # （5） 使用NLTK模块中词干还原方法对样例单词进行词干还原：
    for word in words:
        stemmed_words = [stemmer_porter.stem(word),
                         stemmer_lancaster.stem(word),
                         stemmer_snowball.stem(word)]
        print(formatted_row.format(word, *stemmed_words))
    pass

def nltk_lemmatization():
    # 3.实验实现词型归并
    # （1）导入NLTK中词型归并方法：
    from nltk.stem import WordNetLemmatizer
    # （2）创建样例：
    words = ['table', 'probably', 'wolves', 'playing', 'is',
             'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
    # （3）调用NLTK模块的WordNetLemmatizer()方法：
    lemmatizer_wordnet = WordNetLemmatizer()
    # （4）设置打印输出格式：
    lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
    formatted_row = '{:>24}' * (len(lemmatizers) + 1)
    print('\n', formatted_row.format('WORD', *lemmatizers), '\n')
    # （5）使用NLTK模块中词型归并方法对样例单词进行词型归并：
    for word in words:
        lemmatized_words = [lemmatizer_wordnet.lemmatize(word, pos='n'),
                            lemmatizer_wordnet.lemmatize(word, pos='v')]
        print(formatted_row.format(word, *lemmatized_words))
    pass

def nltk_POS():
    # 4.词性标注 (Part-Of-Speech) nltk.word_tokenize()
    import nltk
    words = nltk.word_tokenize('Python is a widely used programming language.')
    print(nltk.pos_tag(words))  # 需要下载 averaged_perceptron_tagger
    pass

def nltk_stop_words():
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
    words = nltk.word_tokenize('Python is a widely used programming language.')
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    print('原始词：', words)
    print('去除停用词后：', filtered_words)
    pass

# 典型的文本预处理流程
def nltk_demo():
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

def nltk_tfidf():
    """

    创建文本分类器目的是将文档集中的多个文本文档划分为不同的类别，
    文本分类在自然语言处理中是很重要的一种分析手段，
    为实现文本的分类，我们将使用另一种统计数据方法tf-idf
    （词频-逆文档频率）， tf-idf方法与基于单词出现频率的
    统计方法一样，都是将一个文档数据转化为数值型数据的一种方法。
    """
    # （1）导入相关的包：
    from sklearn.datasets import fetch_20newsgroups
    # （2） 创建字典，定义分类类型的列表：
    category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles',
                    'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography', 'sci.space': 'Space'}
    # （3）加载训练数据：
    training_data = fetch_20newsgroups(subset='train',
                                       categories=category_map.keys(), shuffle=True, random_state=7)
    # （4） 特征提取：
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X_train_termcounts = vectorizer.fit_transform(training_data.data)
    print("\nDimensions of training data:", X_train_termcounts.shape)
    # （5）训练分类器模型：
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfTransformer
    # （6）创建随即样例：
    input_data = [
        "The curveballs of right handed pitchers tend to curve to the left",
        "Caesar cipher is an ancient form of encryption",
        "This two-wheeler is really good on slippery roads"]
    # （7）使用tf-idf 算法实现数值型数据的转化以及训练：
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)
    classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)
    # （8）词频于tf-idf作为输入的对比：
    X_input_termcounts = vectorizer.transform(input_data)
    X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)
    # （9）打印输出结果：
    predicted_categories = classifier.predict(X_input_tfidf)
    for sentence, category in zip(input_data, predicted_categories):
        print('\nInput:', sentence, '\nPredicted category:', category_map[training_data.target_names[category]])
    pass

def nltk_NB():
    """
    在自然语言处理中通过姓名识别性别是一项有趣的事情。
    我们算法是通过名字中的最后几个字符以确定其性别。例如，
    如果名字中的最后几个字符是“la”，它很可能是一名女性的名字，
    如“Angela”或“Layla”。相反的，如果名字中的最后几个字符是“im”，
    最有可能的是男性名字，比如“Tim”或“Jim”。
    """
    # （1）导入相关包：
    import random
    from nltk.corpus import names
    from nltk import NaiveBayesClassifier
    from nltk.classify import accuracy as nltk_accuracy
    # （2）定义函数获取性别：
    def gender_features(word, num_letters=2):
        return {'feature': word[-num_letters:].lower()}

    # （3）定义main函数以及数据：
    if __name__ == '__main__':
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                         [(name, 'female') for name in names.words('female.txt')])
        random.seed(7)
        random.shuffle(labeled_names)
        input_names = ['Leonardo', 'Amy', 'Sam']
        # （6）获取末尾字符：
        for i in range(1, 5):
            print('\nNumber of letters:', i)
        featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names]
        # （7）划分训练数据和测试数据：
        train_set, test_set = featuresets[500:], featuresets[:500]
        # （8）分类实现：
        classifier = NaiveBayesClassifier.train(train_set)
        # （9）评测分类效果：
        print('Accuracy ==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))
        for name in input_names:
            print(name, '==>', classifier.classify(gender_features(name, i)))
    pass

def nltk_sentiment_analysis():
    """
    情感分析的实现：
    本文中我们使用NLTK模块中的朴素贝叶斯分类器来实现文档的分类。
    在特征提取函数中，我们提取了所有的词。但是，在此我们注意到
    NLTK分类器的输入数据格式为字典格式，因此，我们下文中创建了
    字典格式的数据，以便我们的NLTK分类器可以使用这些数据。同时，
    在创建完字典型数据后，我们将数据分成训练数据集和测试数据集，
    我们的目的是使用训练数据训练我们的分类器，以便分类器可以将
    数据分为积极与消极。而当我们查看哪些单词包含的信息量最大，
    也就是最能体现其情感的单词的时候，我们会发现有些单词例如，
    “outstanding”表示积极情感，“insulting”表示消极情感。这是
    非常有意义的信息，因为它告诉我们什么单词被用来表明积极。
    """
    # （1）导入相关包：
    import nltk.classify.util
    from nltk.classify import NaiveBayesClassifier
    from nltk.corpus import movie_reviews
    # （2） 定义函数获取情感数据：
    def extract_features(word_list):
        return dict([(word, True) for word in word_list])

    # （3）加载数据，在这里为了方便教学我们使用NLTK自带数据：
    if __name__ == '__main__':
        positive_fileids = movie_reviews.fileids('pos')
        negative_fileids = movie_reviews.fileids('neg')
        # （4）将加载的数据划分为消极和积极：
        features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                              'Positive') for f in positive_fileids]
        features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                              'Negative') for f in negative_fileids]
        # （5）将数据划分为训练数据和测试数据：
        threshold_factor = 0.8
        threshold_positive = int(threshold_factor * len(features_positive))
        threshold_negative = int(threshold_factor * len(features_negative))
        # （6）提取特征：
        features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
        features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
        print("\nNumber of training datapoints:", len(features_train))
        print("Number of test datapoints:", len(features_test))
        # （7）调用朴素贝叶斯分类器：
        classifier = NaiveBayesClassifier.train(features_train)
        print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))
        # （8）输出分类结果：
        print("\nTop 10 most informative words:")
        for item in classifier.most_informative_features()[:10]:
            print(item[0])
        # （9）使用分类器对情感进行预测：
        input_reviews = [
            "It is an amazing movie",
            "This is a dull movie. I would never recommend it to anyone.",
            "The cinematography is pretty great in this movie",
            "The direction was terrible and the story was all over the place"
        ]
        # （10）输出预测的结果：
        print("\nPredictions:")
        for review in input_reviews:
            print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 2))
    pass

if __name__=="__main__":
    # nltk_brown()
    # nltk_tokenize()
    # nltk_stemming()
    # nltk_lemmatization()
    # nltk_POS()
    # nltk_stop_words()
    nltk_demo()
    # nltk_tfidf()
    # nltk_NB()
    # nltk_sentiment_analysis()

    pass
