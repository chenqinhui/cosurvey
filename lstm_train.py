# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import multiprocessing
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml

# set parameters:
vocab_dim = 200
maxlen = 100
batch_size = 32
n_epoch = 4
input_length = 100



#加载训练文件（已分词）
def loadfile():
    negative = pd.read_csv('preProcessFinished/bad.txt', header=None, names=None, lineterminator='\n', encoding='utf-8')
    positive = pd.read_csv('preProcessFinished/good.txt', header=None, names=None, lineterminator='\n', encoding='utf-8')
    neutral = pd.read_csv('preProcessFinished/middle.txt', header=None, names=None, lineterminator='\n', encoding='utf-8')
    # 连接消极、中立、积极三个文件
    combined = np.concatenate(positive.values, neutral.values, negative.values )
    pos_neu = np.append(positive.values, neutral.values)
    pos_neu_neg = np.append(pos_neu, negative.values)
    y = np.concatenate((np.ones(len(neutral.values), dtype=int),
                        np.zeros(len(positive.values), dtype=int),
                        -1*np.ones(len(negative.values), dtype=int)))
    return pos_neu_neg, y


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None, sentences=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (sentences is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}# 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}# 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        sentences=parse_dataset(sentences)
        sentences= sequence.pad_sequences(sentences, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, sentences
    else:
        print('No data provided...')


# 导入word2vec模型
def word2vec_train(sentences):
    model = Word2Vec.load('word2vec/word2vec')
    index_dict, word_vectors, sentences = create_dictionaries(model=model, sentences=sentences)
    return index_dict, word_vectors, sentences

def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    len(combined)
    # train和test数据比例是8:2
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    print(x_train.shape,y_train.shape)
    return n_symbols, embedding_weights, x_train,y_train, x_test, y_test


# 定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    # model.add(LSTM(units=50, activation='tanh'))
    # model.add(Dropout(0.5))
    # model.add(Dense(3,activation='softmax'))
    # model.add(Activation('softmax'))
    model.add(LSTM(300,dropout_U=0.1,dropout_W=0.1))
    model.add(Dense(3, activation='softmax'))

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_model_2/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_model_2/lstm.h5')
    print('Test score:', score)
    print('Test accuracy:', acc)


print('Loading Data...')
sentences, y = loadfile()
print(len(sentences), len(y))
print('Training a Word2vec model...')
index_dict, word_vectors, combined = word2vec_train(sentences)
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
print(x_train.shape, y_train.shape)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
