
# coding: utf-8

# # Quora Kaggle competition
#
# Welcome to the Quora Question Pairs competition! Here, our goal is to identify which questions asked on Quora, a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.

# In[2]:

import numpy as np
import pandas as pd
import os,re
import cPickle
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
import xgboost as xgb
from tqdm import tqdm
from pyemd import emd
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ### Training data

# In[2]:

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')


# In[3]:

print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Total number of question pairs for test data: {}'.format(len(df_test)))
print('Duplicate pairs : {} %'.format(round(df_train['is_duplicate'].mean()*100,2)))


# ## Generate features

# ### Text handcrafted features (fs_1)

# In[4]:

def generate_features(df_train):
    df_train['len_q1'] = df_train['question1'].apply(lambda x:len(str(x)))
    df_train['len_q2'] = df_train['question2'].apply(lambda x:len(str(x)))
    df_train['diff_len'] = df_train.len_q1-df_train.len_q2
    df_train['len_char_q1'] = df_train.question1.apply(lambda x:len(''.join(set(str(x).replace(' ','')))))
    df_train['len_char_q2'] = df_train.question2.apply(lambda x:len(''.join(set(str(x).replace(' ','')))))
    df_train['len_word_q1'] = df_train.question1.apply(lambda x:len(str(x).split()))
    df_train['len_word_q2'] = df_train.question2.apply(lambda x:len(str(x).split()))
    df_train['common_words'] = df_train.apply(lambda x:len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)

    df_train['fuzzy_qratio'] = df_train.apply(lambda x: fuzz.QRatio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_wratio'] = df_train.apply(lambda x:fuzz.WRatio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_partial_ratio'] = df_train.apply(lambda x:fuzz.partial_ratio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_partial_token_set_ratio'] = df_train.apply(lambda x:fuzz.partial_token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_partial_token_sort_ratio'] = df_train.apply(lambda x:fuzz.partial_token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_token_sort_ratio'] = df_train.apply(lambda x:fuzz.token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)
    df_train['fuzzy_token_set_ratio'] = df_train.apply(lambda x:fuzz.token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)
    return df_train


# In[5]:

df_train = generate_features(df_train)
df_test = generate_features(df_test)


# ### LDA (Lattent Dirichlet Allocation) features

# In[6]:

# Steaming
p_stemmer = PorterStemmer()
STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")

    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)

    return data

# Function to vuild a corpus
def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
    return corpus


# In[7]:

data = clean_dataframe(df_train)
corpus = build_corpus(data)
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(text) for text in corpus]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary)


# In[8]:

def common_lda_topic(sentence1,sentence2,dictionary,ldamodel,min_proba):
    "find #common topic based on lattent dirichlet allocation model"
    sentence1 = sentence1.split()
    sentence2 = sentence2.split()

    sentence1 = dictionary.doc2bow(sentence1)
    sentence2 = dictionary.doc2bow(sentence2)

    topic_a = ldamodel.get_document_topics(sentence1,minimum_probability=min_proba)
    topic_b = ldamodel.get_document_topics(sentence2,minimum_probability=min_proba)

    topic_a = list(sorted(topic_a, key=lambda x: x[1]))
    topic_b = list(sorted(topic_b, key=lambda x: x[1]))
    common_topic = set([x[0] for x in topic_a]).intersection(x[0] for x in topic_b)
    return(len(common_topic))


# In[9]:

#vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(vis_data)


# In[10]:

df_train['common_topics'] = df_train.apply(lambda x:common_lda_topic(str(x['question1']),str(x['question2']),dictionary,ldamodel,0.1),axis=1)
df_test['common_topics'] = df_test.apply(lambda x:common_lda_topic(str(x['question1']),str(x['question2']),dictionary,ldamodel,0.1),axis=1)


# ### POS-Tagging features

# In[11]:

def common_pos_tagging(question1,question2):
    question1 = nltk.word_tokenize(question1)
    question2 = nltk.word_tokenize(question2)
    pos_question1 = nltk.pos_tag(question1)
    pos_question2 = nltk.pos_tag(question2)

    pos_1_array = [x[1] for x in pos_question1]
    pos_2_array = [x[1] for x in pos_question2]
    return(len(set(pos_1_array).intersection(pos_2_array)))


# In[12]:

def count_distinct_pos_tagging(df_train):
    #Generate all pos-tag null columns
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    pos_tag = tagdict.keys()
    for tag in pos_tag:
        df_train[tag+"_q1"] = 0
        df_train[tag+"_q2"] = 0

    for index, row in df_train.iterrows():
        question1 = str(row.question1).decode('utf-8')
        question1 = nltk.word_tokenize(question1)

        question2 = str(row.question2).decode('utf-8')
        question2 = nltk.word_tokenize(question2)

        pos_question1 = nltk.pos_tag(question1)
        pos_question1 = [x[1] for x in pos_question1]

        pos_question2 = nltk.pos_tag(question2)
        pos_question2 = [x[1] for x in pos_question2]

        for tag in pos_question1:
            if(tag != "#"):
                df_train.set_value(index,tag+"_q1",row[tag+"_q1"]+1)

        for tag in pos_question2:
            if(tag != "#"):
                df_train.set_value(index,tag+"_q2",row[tag+"_q2"]+1)


# In[13]:

df_train['common_pos_count'] = df_train.apply(lambda x:common_pos_tagging(str(x['question1']).decode('utf-8'),str(x['question2']).decode('utf-8')),axis=1)
df_test['common_pos_count'] = df_test.apply(lambda x:common_pos_tagging(str(x['question1']).decode('utf-8'),str(x['question2']).decode('utf-8')),axis=1)
count_distinct_pos_tagging(df_train)
count_distinct_pos_tagging(df_test)


# In[14]:



# # Word2vec features
# Word2Vec creates a multi-dimensional vector for every word in the english vocabulary (or the corpus it has been trained on). Word2Vec embeddings are very popular in natural language processing and always provide us with great insights. Wikipedia provides a good explanation of what these embeddings are and how they are generated (https://en.wikipedia.org/wiki/Word2vec).

# In[15]:

model = gn.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
model.init_sims(replace=True)


# In[16]:

print("Word2vec mean of president : {} mean of obama {}".format(model['president'].mean(),model['king'].mean()))


# The idea here is to build a vector of each word in differents sentences (question1, question2), each word will be presented in a 300 dimensions vector. So we will try to calculate the average of these vector in order to compare the result obtained in the first and second question. I will try to calculate differents distances based on theses vectors such as euclidiant distance, cosine similiratity, hamming distance ... and generate some features based on semantic of words. So, if sentence $s1$ contains $K$ words, we can present this sentence by $K$ arrays of $300$ dimensions. So a matrix of $[K*300]$.
#
# An example of the application of this model is to find the word the most similar to a particular word (semantic sense I mean). Lets try to find the most similar word to `Obama`.

# In[17]:



# In[18]:

def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = nltk.word_tokenize(words)
    words = [w for w in words if not w in STOP_WORDS]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    return M


# We will take now an example of a sentence and apply the word2vec model of words that contains, Lets take the first observation of the train dataframe. It's `'What is the step by step guide to invest in share market in india?'`, so this sentences contains 7 words (after deleting stop words). So this sentence will be presented by a $[7*300]$ matrix. Let's do that

# In[19]:


# The disadvantage of the `word2vec` model that we can't present all the sentence but only words, so we will try some metodologies to present all the sentence using the `word2vec` model, we will for example try to sum the `word2vec` array and calculate the distance between the 2 questions. So each question will be presented using a ${R}^{300}$ vector, and calculate the distance between two ${R}^{300}$ vectors

# In[20]:



# ### Word mover's distance feature :
# In document classification and other natural language processing applications, having a good measure of the similarity of two texts can be a valuable building block. Ideally, such a measure would capture semantic information. Cosine similarity on bag-of-words vectors is known to do well in practice, but it inherently cannot capture when documents say the same thing in completely different words. Take, for example, two headlines:
#
# <b>`- Obama speaks to the media in Illinois`</b>
#
# <b>`- The President greets the press in Chicago`</b>
#
# These have no content words in common, so according to most bag of words—based metrics, their distance would be maximal. (For such applications, you probably don’t want to count stopwords such as the and in, which don’t truly signal semantic similarity.)
#
#
# The distance between two texts is given by the total amount of “mass” needed to move the words from one side into the other, multiplied by the distance the words need to move. So, starting from a measure of the distance between different words, we can get a principled document-level distance.

# In[33]:

def wmd(s1,s2):
    s1 = str(s1).lower().decode('utf-8').split()
    s2 = str(s2).lower().decode('utf-8').split()
    s1 = [w for w in s1 if w not in STOP_WORDS]
    s2 = [w for w in s2 if w not in STOP_WORDS]
    return model.wmdistance(s1,s2)


# In[24]:

df_train['wmd'] = df_train.apply(lambda x:model.wmdistance(str(x['question1']),str(x['question2'])),axis=1)
df_test['wmd'] = df_test.apply(lambda x:model.wmdistance(str(x['question1'],x['question2'])),axis=1)


# In[32]:



# ### Calculcate differents distances (euclidian, cosine, jaccard, monkowski ...)
# We will try now to calculate the distance between two questions. i.e between the two ${R}^{300}$ vector, so we will present these two vector as 2 points in a plan of $300$ dimensions, and calculate the distance between those points. Let's do that

# In[39]:

def word2vec_sentences(s):
    M = sent2vec(s)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


# In[66]:

# Array contains the word2vec sum (sentences)
question1_vectors = np.zeros((df_train.shape[0], 300))
question2_vectors = np.zeros((df_train.shape[0], 300))

question1_test_vectors = np.zeros((df_train.shape[0], 300))
question2_test_vectors = np.zeros((df_train.shape[0], 300))
for i, q in tqdm(enumerate(df_train.question1.values)):
    question1_vectors[i, :] = word2vec_sentences(q)

for i,q in tqdm(enumerate(df_train.question2.values)):
    question2_vectors[i, :] = word2vec_sentences(q)

for i, q in tqdm(enumerate(df_test.question1.values)):
    question1_test_vectors[i, :] = word2vec_sentences(q)

for i,q in tqdm(enumerate(df_test.question2.values)):
    question2_test_vectors[i, :] = word2vec_sentences(q)


# In[98]:

df_train['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_train['euclidean_distance'] = [euclidean(x, y) for(x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]

df_test['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]
df_test['euclidean_distance'] = [euclidean(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['jaccard_distance'] = [jaccard(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['canberra_distance'] = [canberra(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['minkowski_distance'] = [minkowski(x, y, 3) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['braycurtis_distance'] = [braycurtis(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['canberra_distance'] = [canberra(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]


df_train['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
df_test['cityblock_distance'] = [cityblock(x, y) for(x, y) in zip(np.nan_to_num(question1_test_vectors),np.nan_to_num(question2_test_vectors))]



df_train.to_csv('train_w2v_features.csv', index=False)
df_test.to_csv('test_w2v_features.csv', index=False)

# In[99]:





# Set our parameters for xgboost
from sklearn.metrics import accuracy_score

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

x_train = df_train.ix[:, 6:,]
y_train = df_train.is_duplicate

d_train = xgb.DMatrix(x_train, label=y_train)


bst = xgb.train(params, d_train, 400)

test = df_test.ix[:, 3:,]
d_test = xgb.DMatrix(test)
p_test = bst.predict(d_test)


sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('complicated_xgb.csv', index=False)
