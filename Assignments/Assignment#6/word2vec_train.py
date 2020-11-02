from gensim.models.keyedvectors import KeyedVectors
import time
import datetime
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
from settings import settings

#-------------pre-trained word2vec---------------
model = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True, limit=60000)
score, predictions = model.evaluate_word_analogies('./data/questions-words.txt')

print(score)
print(model['apple'])
print("similarity between apple and fruit: {}".format(model.similarity("apple", "fruit")))
print("similarity between apple and car: {}".format(model.similarity("apple", "car")))
print(model.most_similar("apple", topn=10))
print(model.most_similar(positive=['king', 'women'], negative=['man'], topn=10))

-------------training---------------

setting = settings['SET#4']

start_time = time.time()

sentences = PathLineSentences("./data/1billion/")
model = Word2Vec(sentences, size=setting['DIMENSION'], window=5, min_count=5, workers=4, sg=setting['SG'], hs=setting['HS'],
                 negative=setting['NEGATIVE'], ns_exponent=0.75, cbow_mean=1, alpha=setting['ALPHA'],
                 min_alpha=setting['MIN_ALPHA'], iter=setting['ITER'])
model.save("./set4/word2vec.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)
print(f"training_time: {(time.time() - start_time) / 60}")

# -------------evaluation---------------
# model = Word2Vec.load("./set2/word2vec.model")
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)
#
# print(model.wv.most_similar("car", topn=200))
# print(len(model.wv.vocab))
# print("similarity between apple and fruit: {}".format(model.wv.similarity("apple", "fruit")))
# print(model.wv["apple"])


