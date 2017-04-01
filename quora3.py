
# coding: utf-8

# ## Quora Kaggle competition
#
# Welcome to the Quora Question Pairs competition! Here, our goal is to identify which questions asked on Quora, a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.

# In[66]:

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
    stop_words = stop_words.word('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1,s2)




def norm_wmd(s1,s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stop_words.word('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


# ### Training data

# In[38]:

df_train = pd.read_csv('./train_features.csv')
df_test = pd.read_csv('./test_features.csv')

# Generate word2vec features

#Load Word2vec models trained on Google news corpus (300 dimensions)
model = gn.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)

#Calculate word mover distance based on word2vec model
df_train['wmd'] = df_train.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)





# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

x_train = df_train.ix[:, 6:24]
y_train = df_train.is_duplicate

d_train = xgb.DMatrix(x_train, label=y_train)


bst = xgb.train(params, d_train, 400)

print(bst.feature_names)

test = df_test.ix[:, 3:21]
d_test = xgb.DMatrix(test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb_2.csv', index=False)
