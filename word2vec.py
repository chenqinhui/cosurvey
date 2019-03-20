# import modules & set up logging
from gensim.models import word2vec
import logging
import multiprocessing
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = time.time()
# 导入分词后的评论语料
# 取频率高于5的词，维度为200，窗口大小为5
print('start running.')
sentences = word2vec.Text8Corpus(u"preProcessFinished/comments.txt")
model = word2vec.Word2Vec(sentences, min_count=5, size=200, workers=multiprocessing.cpu_count(), window=5)
model.save('word2vec/word2vec')
model.wv.save_word2vec_format('word2vec/word_vectors', 'word2vec/vocabulary', binary=False)
print('end running.')
end = time.time()
print('running time:'+str(end-start))
