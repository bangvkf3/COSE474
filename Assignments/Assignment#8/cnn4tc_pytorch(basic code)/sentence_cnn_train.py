import torch
import numpy as np
import time
import random
import re
import smart_open
import pickle
import os
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from DNN.COSE474.sentence_cnn import SentenceCnn, Config
import DNN.COSE474.data_helpers as dh

from gensim.models.keyedvectors import KeyedVectors


if __name__ == '__main__':
    # configuration
    cfg = Config()

    print('[CNN for sentence classification training]')
    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 데이터 로드
    if cfg.task == "MR":
        x_text, y = dh.load_mr_data(cfg.mr_train_file_pos, cfg.mr_train_file_neg)
        cfg.nb_classes = max(y) + 1
        print("cfg.nb_classes: ", cfg.nb_classes)
    elif cfg.task == "TREC":
        pass

    word_id_dict, _ = dh.buildVocab(x_text, cfg.vocab_size)  # training corpus를 토대로 단어사전 구축
    cfg.vocab_size = len(word_id_dict) + 4  # 30000 + 4
    print("vocabulary size: ", cfg.vocab_size)

    for word in word_id_dict.keys():
        word_id_dict[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
    word_id_dict['<pad>'] = 0  # zero padding을 위한 토큰
    word_id_dict['<unk>'] = 1  # OOV word를 위한 토큰
    word_id_dict['<s>'] = 2  # 문장 시작을 알리는 start 토큰
    word_id_dict['</s>'] = 3  # 문장 마침을 알리는 end 토큰

    x_indices = dh.text_to_indices(x_text, word_id_dict, True)
    data = list(zip(x_indices, y))
    random.shuffle(data)
    x_indices, y = zip(*data)

    dev_sample_index = -1 * int(cfg.dev_sample_percentage * float(len(y)))
    train_x, dev_x = x_indices[:dev_sample_index], x_indices[dev_sample_index:]
    train_y, dev_y = y[:dev_sample_index], y[dev_sample_index:]

    # data 개수 확인
    print('The number of training data: ', len(train_x))
    print('The number of validation data: ', len(dev_x))

    if cfg.word2vec:  # word2vec 활용 시
        print("Loading W2V data...")
        pre_emb = KeyedVectors.load_word2vec_format(cfg.word2vec, binary=True)  # pre-trained word2vec load
        pre_emb.init_sims(replace=True)
        num_keys = len(pre_emb.vocab)
        print("loaded word2vec len ", num_keys)

        # initial matrix with random uniform, pretrained word2vec으로 vocabulary 내 단어들을 초기화하기 위핸 weight matrix 초기화
        initW = np.random.uniform(-0.25, 0.25, (cfg.vocab_size, cfg.embedding_dim))
        # load any vectors from the word2vec
        print("init initW cnn.W in FLAG")
        for w in word_id_dict.keys():
            arr = []
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in pre_emb:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                arr = pre_emb[w]  # word2vec vector를 가져옴
            elif w.lower() in pre_emb:  # 소문자로도 확인
                arr = pre_emb[w.lower()]
            elif s in pre_emb:  # 전처리 후 확인
                arr = pre_emb[s]
            elif s.isdigit():  # 숫자이면
                arr = pre_emb['1']
            if len(arr) > 0:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                idx = word_id_dict[w]  # 단어 index
                initW[idx] = np.asarray(arr).astype(np.float32)  # 적절한 index에 word2vec word 할당
            initW[0] = np.zeros(cfg.embedding_dim)

    nb_pad = int(max(cfg.filter_lengths) / 2 + 0.5)

    # - 학습 데이터 배치 만들기
    train_x = dh.sequence_to_tensor(train_x, nb_paddings=(nb_pad, nb_pad))
    train_y = torch.tensor(train_y)
    training_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # - 데브 배치 만들기
    dev_x = dh.sequence_to_tensor(dev_x, nb_paddings=(nb_pad, nb_pad))
    dev_y = torch.tensor(dev_y)
    dev_loader = DataLoader(TensorDataset(dev_x, dev_y), batch_size=cfg.batch_size, shuffle=False)

    with smart_open.smart_open("saved_model/vocab", 'wb') as f:
        pickle.dump(word_id_dict, f)
    with smart_open.smart_open("saved_model/emb", 'wb') as f:
        pickle.dump(initW, f)

    # 학습 모델 생성
    model = SentenceCnn(nb_classes=cfg.nb_classes,
                        word_embedding_numpy=initW,
                        filter_lengths=cfg.filter_lengths,
                        filter_counts=cfg.filter_counts,
                        dropout_rate=cfg.dropout_rate).to(device)


    if torch.cuda.is_available():
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay) # lamda 값과 함께 weight decay 적용
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, gamma=0.99, step_size=1000) # 1000 step 마다 lr을 99%로 감소시킴

    # training 시작
    start_time = time.time()
    highest_val_acc = 0
    val_acc_list = []
    global_steps = 0
    print('========================================')
    print("Start training...")
    for epoch in range(cfg.max_epoch):
        train_loss = 0
        train_batch_cnt = 0
        model.train()
        for x, y in training_loader:
            global_steps += 1

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()# iteration 마다 gradient를 0으로 초기화
            outputs = model(x)
            loss = criterion(outputs, y)#cross entropy loss 계산
            loss.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트
            step_lr_scheduler.step(global_steps) # learning rate 업데이트

            train_loss += loss
            train_batch_cnt += 1


        ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_lr()[0])

        # validation (for early stopping)
        correct_cnt = 0
        model.eval()
        for x, y in dev_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model.forward(x)
            _, top_pred = torch.topk(pred, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            correct_cnt += int(torch.sum(top_pred == y))# 맞춘 개수 카운트

        val_acc = correct_cnt / len(dev_y) * 100
        print("validation dataset accuracy: %.2f" % val_acc)
        val_acc_list.append(val_acc)
        if val_acc > highest_val_acc:# validation accuracy가 경신될 때
            save_path = './saved_model/setting_1/epoch_' + str(epoch + 1) + '.pth'
            # 위와 같이 저장 위치를 바꾸어 가며 각 setting의 epoch마다의 state를 저장할 것.
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)# best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            highest_val_acc = val_acc
        epoch += 1

    epoch_list = [i for i in range(1, epoch + 1)]
    plt.title('Validation dataset accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_list, val_acc_list)
    plt.show()
