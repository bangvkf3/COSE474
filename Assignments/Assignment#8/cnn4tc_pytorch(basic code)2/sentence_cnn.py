import numpy as np
import torch
from torch import nn

cpu = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else cpu

class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        conv = []
        for size, num in zip(filter_lengths, filter_counts): #filter size 별로 초기화
            conv2d = nn.Conv2d(1, num, (size, word_dimension)) # (input_channel, ouput_channel, height, width)
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu') # He initialization
            nn.init.zeros_(conv2d.bias)
            conv.append(nn.Sequential(
                conv2d,
                nn.ReLU(inplace=True)
            ))

        self.conv = nn.ModuleList(conv)
        self.filter_sizes = filter_lengths
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):
        features = []
        for filter_size, conv in zip(self.filter_sizes, self.conv): #filter size 별로 convolution 수행
            # embedded_words: [batch, sentence length, embedding dimension]
            conv_output = conv(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]  # max over-time pooling
            features.append(conv_output)
            del conv_output

        features = torch.cat(features, dim=1) # 각각의 filter에서 나온 feature들을 concatenation
        dropped_features = self.dropout(features)
        return dropped_features


class SentenceCnn(nn.Module):
    def __init__(self, nb_classes, word_embedding_numpy, filter_lengths, filter_counts, dropout_rate):
        super().__init__()

        vocab_size = word_embedding_numpy.shape[0]
        word_dimension = word_embedding_numpy.shape[1]

        # 워드 임베딩 레이어
        self.word_embedding = nn.Embedding(
            vocab_size,
            word_dimension,
            padding_idx=0
        ).to(device)
        
        # word2vec 활용
        self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))

        # 컨볼루션 레이어
        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)

        # 풀리 커텍티드 레이어
        nb_total_filters = sum(filter_counts)
        self.linear = nn.Linear(nb_total_filters, nb_classes).to(device)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        x = self.word_embedding(input_x).to(device)
        x = x.unsqueeze(1)  # 채널 1개 추가
        x = self.features(x)
        logits = self.linear(x)
        return logits

class Config():
    def __init__(self):
        self.batch_size = 50
        self.lr = 0.1
        self.dropout_rate = 0.5
        self.weight_decay = 1e-04
        self.filter_lengths = [3, 4, 4, 5]
        self.filter_counts = [128, 128, 128]
        self.nb_classes = 0
        self.embedding_dim = 300
        self.vocab_size = 30000
        self.dev_sample_percentage = 0.1
        self.max_epoch = 120
        self.task = "MR"
        self.mr_train_file_pos = "./data/MR/rt-polarity.pos"
        self.mr_train_file_neg = "./data/MR/rt-polarity.neg"
        self.trec_train_file = "./data/TREC/traindata.txt"
        self.word2vec = "./data/GoogleNews-vectors-negative300.bin" # or None

