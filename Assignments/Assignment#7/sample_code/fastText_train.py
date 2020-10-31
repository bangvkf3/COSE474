from gensim.models import FastText
from gensim.models.word2vec import PathLineSentences

sentences = PathLineSentences("./data/1billion/")
model = FastText(sentences=sentences, size=100, window=5, min_count=10, workers=4, sg=0, hs=0,
                  negative=5, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=1,
                 word_ngrams=1, min_n=3, max_n=6)
model.save("fastText.model")
print(len(model.wv.vocab))
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)

# model = FastText.load("fastText.model")
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)
# print(model.wv.most_similar("thank____you", topn=20))
# print(len(model.wv.vocab))


