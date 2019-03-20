import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

import pandas as pd
import json
import yaml
from keras.models import model_from_yaml

np.random.seed(1337)  # For Reproducibility
import sys

sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100


def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined):  # 闭包-->临时使用
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freqxiao10->0
                data.append(new_txt)
            return data  # word=>index

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('word2vec/word2vec')
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open('lstm_model_2/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('lstm_model_2/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    # print data
    result = model.predict_classes(data)
    # print result # [[1]]
    if result[0] == 1:
        print(string, ' positive')
        return 1
    elif result[0] == 0:
        print(string, ' neural')
        return 0
    else:
        print(string, ' negative')
        return -1

# 去重
# def remove_duplicated(comment_list):
#     out_file = open("commentData/sn_bad_3.jsonlines",'a',encoding='utf-8')
#     # columns = ['content']
#     content_list = []
#     for comment in comment_list:
#         content_list.append(comment)
#     dic = {'content': content_list}
#     frame = pd.DataFrame(dic)
#     new_frame = frame.drop_duplicates('content')
#     i = 0
#     for indexs in new_frame.index:
#         item = new_frame.loc[indexs].values
#         out = item[0]
#         out_file.write(out+'\n')
#         i = i+1
#     print(str(i)+" removed duplicated.")
#     out_file.close()

def select_comment(comment_list):
    # columns = ['good_id', 'content', 'score']
    good_id_list = []
    content_list = []
    for comment in comment_list:
        # items = comment.split('||')
        comment_json = json.loads(comment)
        good_id_list.append(comment_json['good_id']+'-'+comment_json['shop_id'])
        content_list.append(comment_json['comment']['content'])
    dic = {'good_id': good_id_list, 'content': content_list}
    frame = pd.DataFrame(dic)
    # new_frame = frame['content'].groupby(frame['good_id'])

    frame = frame.groupby(['good_id'])
    for group in frame:
        pos = 0
        neg = 0
        neu = 0
        df1 = group[1]['content'] # 类型为Series
        for content in df1:
            if content!='此用户没有填写评价内容' and content!='买家没有填写评价内容！':
                result = lstm_predict(content)
                if result == 1:
                    pos+=1
                elif result ==0:
                    neu+=1
                else:
                    neg+=1
        print("good_id:",group[0], "positive:", pos, "neural:", neu, "negative:", neg)



if __name__ == '__main__':
    good_filename = "commentData/sn_good_3.jsonlines"
    good_file = open(good_filename,'r',encoding='utf-8')
    middle_filename = "commentData/sn_middle_3.jsonlines"
    middle_file = open(middle_filename, 'r',encoding='utf-8')
    bad_filename = "commentData/sn_bad_3.jsonlines"
    bad_file = open(bad_filename, 'r',encoding='utf-8')
    comments = []
    for line in good_file.readlines():
        comments.append(line[:-1])
    good_file.close()
    for line in middle_file.readlines():
        comments.append(line[:-1])
    middle_file.close()
    for line in bad_file.readlines():
        comments.append(line[:-1])
    bad_file.close()
    # remove_duplicated(comments)
    select_comment(comments)
    # strings = ["电脑质量太差了，傻逼店家，赚黑心钱，以后再也不会买了",
    #            "不是太好",
    #            "不错不错",
    #            "物流很快 运行流畅 值得信赖的大品牌",
    #            "最差的卖家，态度恶略，因为对京东的信任购买，没想到平台上这么差劲?的卖家。失望",
    #            "真的一般"]
    # for string in strings:
    #     lstm_predict(string)