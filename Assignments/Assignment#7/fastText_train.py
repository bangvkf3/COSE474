import time
from gensim.models import FastText
from gensim.models.word2vec import PathLineSentences
from settings import settings
import os

# f = open("./train_result.txt", 'w')  # result 파일 생성

# results = dict()  # 결과 시트
#
# set_start = 3
# set_end = 3
#
# for i in range(set_start, set_end + 1):
#     setting = settings[f'SET#{i}']
#     if not os.path.exists(f'set{i}/'):
#         os.makedirs(f'set{i}/')
#     start_time = time.time()
#     sentences = PathLineSentences("./data/1billion/")
#     model = FastText(sentences=sentences, size=setting['DIMENSION'], window=5, min_count=10, workers=4, sg=setting['SG'],
#                      hs=setting['HS'], negative=setting['NEGATIVE'], ns_exponent=0.75, alpha=setting['ALPHA'],
#                      min_alpha=setting['MIN_ALPHA'], iter=setting['ITER'], word_ngrams=1, min_n=setting['MIN_N'],
#                      max_n=setting['MAX_N'])
#     model.save(f"./set{i}/fastText.model")
#     result = dict()
#     result['len'] = len(model.wv.vocab)
#     score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
#     result['training_time'] = (time.time() - start_time) / 60
#     result['score'] = score
#     results[f'Setting#{i}'] = result
#
# for setting, result in results.items():
#     f.write(f"< {setting} result >\n")
#     f.write(f" - Training time: {result['training_time']}\n")
#     f.write(f" - Score : {result['score']}\n")
#     f.write(f" - len of model.wv.vocab : {result['len']}\n\n")

# f.close()  # result 파일 닫기

#####################################################################################

model = FastText.load("./set3/fastText.model")
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)
print(model.wv.most_similar("braavo", topn=20))
print(len(model.wv.vocab))


