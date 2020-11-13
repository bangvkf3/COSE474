import numpy as np
import re
import collections

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_trec_data(train_file):
    # Load data from files
    pass

def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    pos_text = [clean_str(sent) for sent in pos_text] # clean_str 함수로 전처리 (소문자, 특수 기호 제거, (), 등 분리)


    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines()) # 부정적인 review 읽어서 list 형태로 관리
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [[0, 1] for _ in pos_text] # 긍정 review 개수만큼 ground_truth 생성 [0, 1]
    negative_labels = [[1, 0] for _ in neg_text] # 부정 review 개수만큼 ground_truth 생성 [0, 1]
    y = np.concatenate([positive_labels, negative_labels], 0)

    x_final = pos_text + neg_text
    return [x_final, y]

def buildVocab(sentences, vocab_size):
    # Build vocabulary
    words = []
    for sentence in sentences: words.extend(sentence.split()) # i, am, a, boy, you, are, a, girl
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # a: 0, i: 1...
    return [vocabulary, vocabulary_inv]

def text_to_index(text_list, word_to_id, nb_pad):
    text_indices = []
    for text in text_list:
        words = text.split(" ")
        pad = [0 for _ in range(nb_pad) ] # zero padding
        ids = []
        for word in words: # i, am, a, boy
            if word in word_to_id:
                word_id = word_to_id[word]
            else:
                word_id = 1 # OOV (out-of-vocabulary)
            ids.append(word_id) # 5, 8, 6, 19
        ids = pad + ids # 0, 0, 0, 0, 5, 8, 6, 19
        text_indices.append(ids)
    return text_indices

def train_tensor(batches):
    max_length = max([len(batch) for batch in batches]) # 100
    tensor = np.zeros((len(batches), max_length), dtype=np.int64) #(5000, 100)
    # 0 0 0 0 5 8 6 19 0 0....0 0 0
    # 0 0 0 0 5 7 11 1 1 1....1 1 1
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    #...
    # 0 0 0 0 0 0 0 0 0 0....0 0 0
    for i, indices in enumerate(batches):
        tensor[i, :len(indices)] = np.asarray(indices, dtype=np.int64)
    return tensor, max_length

def test_tensor(batches, max_length):
    tensor = np.zeros((len(batches), max_length), dtype=np.int64)
    for i, indices in enumerate(batches):
        if len(indices) > max_length:
            tensor[i, :max_length] = np.asarray(indices[:max_length], dtype=np.int64)
        else:
            tensor[i, :len(indices)] = np.asarray(indices, dtype=np.int64)

    return tensor

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]