
# coding: utf-8


import numpy as np
import pandas as pd
import os,re
import seaborn as sns
import gensim as gn
import logging
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.models.word2vec import Word2Vec
import nltk
import scipy.sparse as sparse
from nltk.data import load
from fuzzywuzzy import fuzz
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
import xgboost as xgb


# WMD (Word Mover Distance) :
def wmd(s1,s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = nltk.corpus.stopwords.words()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1,s2)




def norm_wmd(s1,s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = nltk.corpus.stopwords.words()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


# ### Training data

# In[38]:

df_train = pd.read_csv('./train_features.csv')
df_test = pd.read_csv('./test_features.csv')

# Generate word2vec features

#Load Word2vec models trained on Google news corpus (300 dimensions)
model = gn.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

#Calculate word mover distance based on word2vec model
df_train['wmd'] = df_train.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
df_test['wmd'] = df_test.apply(lambda x: wmd(x['question1'],x['question2']), axis=1)


#Calculate normalised word mover distance based on word2vec model
df_train['norm_wmd'] = df_train.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
df_test['norm_wmd'] = df_test.apply(lambda x: norm_wmd(x['question1'],x['question2']), axis=1)



# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

x_train = df_train.ix[:, 6:,]
y_train = df_train.is_duplicate

d_train = xgb.DMatrix(x_train, label=y_train)


bst = xgb.train(params, d_train, 400)

print(bst.feature_names)

test = df_test.ix[:, 3:,]
d_test = xgb.DMatrix(test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb_2.csv', index=False)
