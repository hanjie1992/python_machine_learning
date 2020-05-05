import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
from datetime import datetime
import time
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./data/train_set.csv')
print(data.head())
print(data.columns)

data_x = data.loc[:,data.columns != 'label']
data_y = data.loc[:,data.columns == 'label']

print(data_x.head())
print(data_y.head())

x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size = 0.2, random_state = 0)
print(x_test.shape)

x_val = x_test.iloc[:1500,:]
y_val = y_test.iloc[:1500,:]

x_test = x_test.iloc[1500:,:]
y_test = y_test.iloc[1500:,:]

print (x_val.shape)
print (x_test.shape)

del x_train['user_id']
del x_train['sku_id']

del x_val['user_id']
del x_val['sku_id']

print(x_train.head())

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_val, label=y_val)

param = {'n_estimators': 4000, 'max_depth': 3, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0,
             'colsample_bytree': 0.8, 'scale_pos_weight':10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic',
             'eval_metric':'auc'}

num_round = param['n_estimators']

plst = param.items()
evallist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
bst.save_model('bst.model')

print (bst.attributes())

def create_feature_map(features):
    outfile = open(r'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


features = list(x_train.columns[:])
create_feature_map(features)

def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore(fmap=r'xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    file_name = './data/feature_importance_' + str(datetime.now().date())[5:] + '.csv'
    df.to_csv(file_name)

feature_importance(bst)

fi = pd.read_csv('./data/feature_importance_10-24.csv')
fi.sort_values("fscore", inplace=True, ascending=False)
print(fi.head())

print(x_test.head())

users = x_test[['user_id', 'sku_id', 'cate']].copy()
del x_test['user_id']
del x_test['sku_id']
x_test_DMatrix = xgb.DMatrix(x_test)
y_pred = bst.predict(x_test_DMatrix, ntree_limit=bst.best_ntree_limit)

x_test['pred_label'] = y_pred
print(x_test.head())

def label(column):
    if column['pred_label'] > 0.5:
        #rint ('yes')
        column['pred_label'] = 1
    else:
        column['pred_label'] = 0
    return column
x_test = x_test.apply(label,axis = 1)
print(x_test.head())

x_test['true_label'] = y_test
print(x_test.head())

#x_test users = x_test[['user_id', 'sku_id', 'cate']].copy()
x_test['user_id'] = users['user_id']
x_test['sku_id'] = users['sku_id']
print(x_test.head())

# 所有购买用户
all_user_set = x_test[x_test['true_label']==1]['user_id'].unique()
print (len(all_user_set))
# 所有预测购买的用户
all_user_test_set = x_test[x_test['pred_label'] == 1]['user_id'].unique()
print (len(all_user_test_set))
all_user_test_item_pair = x_test[x_test['pred_label'] == 1]['user_id'].map(str) + '-' + x_test[x_test['pred_label'] == 1]['sku_id'].map(str)
all_user_test_item_pair = np.array(all_user_test_item_pair)
print (len(all_user_test_item_pair))
#print (all_user_test_item_pair)

pos, neg = 0,0
for user_id in all_user_test_set:
    if user_id in all_user_set:
        pos += 1
    else:
        neg += 1
all_user_acc = 1.0 * pos / ( pos + neg)
all_user_recall = 1.0 * pos / len(all_user_set)
print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
print ('所有用户中预测购买用户的召回率' + str(all_user_recall))

#所有实际商品对
all_user_item_pair = x_test[x_test['true_label']==1]['user_id'].map(str) + '-' + x_test[x_test['true_label']==1]['sku_id'].map(str)
all_user_item_pair = np.array(all_user_item_pair)
#print (len(all_user_item_pair))
#print(all_user_item_pair)
pos, neg = 0, 0
for user_item_pair in all_user_test_item_pair:
    #print (user_item_pair)
    if user_item_pair in all_user_item_pair:
        pos += 1
    else:
        neg += 1
all_item_acc = 1.0 * pos / ( pos + neg)
all_item_recall = 1.0 * pos / len(all_user_item_pair)
print ('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
score = 0.4 * F11 + 0.6 * F12
print ('F11=' + str(F11))
print ('F12=' + str(F12))
print ('score=' + str(score))