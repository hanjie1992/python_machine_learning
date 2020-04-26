

def apriori():
    """
    使用Apriori算法找出数据的频繁项集，进而分析物品关联度
    :return:
    """

    #导包
    import pandas as pd
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori

    #设置数据集
    data_set = [['牛奶','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
                ['莳萝','洋葱','肉豆蔻','芸豆','鸡蛋','酸奶'],
                ['牛奶','苹果','芸豆','鸡蛋'],
                ['牛奶','独角兽','玉米','芸豆','酸奶'],
                ['玉米','洋葱','洋葱','芸豆','冰淇淋','鸡蛋']]

    te = TransactionEncoder()
    #进行one-hot编码
    te_ary = te.fit(data_set).transform(data_set)
    print(type(te_ary))
    df = pd.DataFrame(te_ary,columns=te.columns_)
    #利用apriori找出频繁项集
    freq = apriori(df,min_support=0.4,use_colnames=True)

    #导入关联规则包
    from mlxtend.frequent_patterns import association_rules
    #计算关联规则
    result = association_rules(freq,metric="confidence",min_threshold=0.6)
    # 排序
    result.sort_values(by = 'confidence',ascending=False,axis=0)
    print(result)
    result.to_excel("./result.xlsx")
    return None


if __name__=="__main__":
    """
    无监督学习算法
    推荐算法:又叫亲和性分析、关联规则
    简单点说，就是先找频繁项集，再根据关联规则找关联物品。
    Apriori算法进行物品的关联分析
        支持度(support)：可以理解为物品的流行程度。支持度=(包含物品A的记录数)/(总的记录数)
        置信度(confidence):置信度是指如果购买物品A，有多大可能购买物品B。置信度(A->B)=(包含物品A和B的记录数)/(包含A的记录数)
        提升度(lift):指当销售一个物品时，另一个物品销售率会增加多少。提升度(A->B)=置信度(A->B)/(支持度A)。
            提升度大于1，说明A物品卖的越多，B也会卖的越多。等于1 说明A、B之间没有关联。小于1,说明A卖的越多反而建设B卖的数量
        
        问题：
            顾客   购买商品集合
            顾客1：牛奶,洋葱,肉豆蔻,芸豆,鸡蛋,酸奶
            顾客2：莳萝,洋葱,肉豆蔻,芸豆,鸡蛋,酸奶
            顾客3：牛奶,苹果,芸豆,鸡蛋
            顾客4：牛奶,独角兽,玉米,芸豆,酸奶
            顾客5：玉米,洋葱,洋葱,芸豆,冰淇淋,鸡蛋
            
        支持度：(包含物品A的记录数)/(总的记录数)
            牛奶 = 3/5
            鸡蛋 = 4/5
        置信度：(包含物品A和B的记录数)/(包含A的记录数)
            {鸡蛋，牛奶} = 2次
            {牛奶}=3次
            confidence{鸡蛋，牛奶} = 2/3
            购买牛奶的顾客中，有2/3的顾客会购买鸡蛋
        提升度：置信度(A->B)/(支持度A)。
             {鸡蛋，牛奶}的置信度为：2/3
             牛奶的支持度为：3/5
             提升度为：(2/3)/(3/5)=1.11
             说明牛奶卖的越多，鸡蛋也会增多
    """
    apriori()
