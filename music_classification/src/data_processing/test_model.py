# -*- coding: utf-8 -*-
import operator
import os
import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Kkma

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

kkma = Kkma()

emotion_list = []                 # 감정 사전: 감정 단어가 담길 리스트 [모델의 input]

# sad_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_sad_12.model')
happy_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_happy_01.model')
# angry_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_angry_04.model')
# neutral_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_neutral_01.model')

# sad_emotion = ['사랑', '슬픔', '분노', '무관심']
# happy_emotion = ['사랑', '슬픔', '긍정', '행복']
# angry_emotion = ['시발', '슬픔', '분노', '행복']

# sad_emotion = ['이별', '사랑', '슬픔', '눈물', '분노', '무관심']
# happy_emotion = ['사랑', '행복', '예쁘', '미소', '자유']
# angry_emotion = ['시발', '새끼', '바보', '욕', '짜증']
# neutral_emotion = ['힘']

sad_emotion = ['슬픔', '후회', '그리움', '외로움', '무관심', '분노']
happy_emotion = ['기대', '자신감', '긍정', '기쁨', '행복', '감동', '사랑']
# angry_emotion = ['무서움', '두려움', '실망', '분노']
# neutral_emotion = ['힘', '지루함']

w2v_emotion_path = 'w2v_emotion_words'

# print('sad')
# for index in range(len(sad_emotion)):
#     emotion_one = []
#     print(sad_emotion[index])
#     fw = open(w2v_emotion_path + '/w2v_sad_04.txt', 'a', encoding='utf8')
#     for word in sad_model.wv.most_similar(sad_emotion[index], topn=1500):
#         print(word)
#         if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
#             emotion_one.append(word)
#             fw.write(word[0] + ', ' + str(word[1]) + '\n')
#     if emotion_one:
#         emotion_list.append(emotion_one)
#     print()
# print()
# fw.close()


print('happy')
for index in range(len(happy_emotion)):
    emotion_one = []
    print(happy_emotion[index])
    fw = open(w2v_emotion_path + '/w2v_happy_01.txt', 'a', encoding='utf8')
    for word in happy_model.wv.most_similar(happy_emotion[index], topn=1500):
        print(word)
        if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
            emotion_one.append(word)
            fw.write(word[0] + ', ' + str(word[1]) + '\n')
    if emotion_one:
        emotion_list.append(emotion_one)
    print()
print()
fw.close()

#
# print('angry')
# for index in range(len(angry_emotion)):
#     emotion_one = []
#     print(angry_emotion[index])
#     for word in angry_model.wv.most_similar(angry_emotion[index], topn=100):
#         print(word)
#         if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
#             emotion_one.append(word)
#     if emotion_one:
#         emotion_list.append(emotion_one)
#     print()
# print()
#
# print('neutral')
# for index in range(len(neutral_emotion)):
#     emotion_one = []
#     print(neutral_emotion[index])
#     for word in neutral_model.wv.most_similar(neutral_emotion[index], topn=100):
#         print(word)
#         if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
#             emotion_one.append(word)
#     if emotion_one:
#         emotion_list.append(emotion_one)
#     print()
print()
print("emotion_list", emotion_list)
print('총 테스트의 개수: {}'.format(len(emotion_list)))

#
# # 원활한 계산을 위한 dictionary 구성
# sad_data_dic = {sad_emotion[0]: 0, sad_emotion[1]: 0, sad_emotion[2]: 0, sad_emotion[3]: 0}
# happy_data_dic = {happy_emotion[0]: 0, happy_emotion[1]: 0, happy_emotion[2]: 0, happy_emotion[3]: 0}
# angry_data_dic = {angry_emotion[0]: 0, angry_emotion[1]: 0, angry_emotion[2]: 0, angry_emotion[3]: 0}
# # 테스트 예시
# sentence = '7년을 만났죠아무도 우리가이렇게 쉽게 이별할 줄은 몰랐죠그래도 우리는헤어져 버렸죠긴 시간 쌓아왔던기억을 남긴채우린 어쩜 너무 어린 나이에서로를 만나 기댔는지 몰라변해가는 우리 모습들을감당하기 어려웠는지도이별하면 아프다고 하던데그런 것도 느낄 수가 없었죠그저 그냥 그런가봐 하며담담했는데울었죠시간이 가면서 내게 준아쉬움에 그리움에내뜻과는 다른나의 맘을 보면서처음엔 친구로다음에는 연인사이로헤어지면 가까스로친구사이라는그 말 정말 맞는데그 후로 3년을보내는 동안에도가끔씩 서로에게연락을 했었죠다른 한 사람을 만나 또다시사랑하게 되었으면서도 난슬플때면 항상 전활 걸어소리없이 눈물만 흘리고너도 좋은 사람 만나야된다마음에도 없는 말을 하면서아직 나를 좋아하나괜히 돌려 말했죠알아요서로 가장 순수했었던그때 그런 사랑다시 할 수 없다는 걸추억으로 남을 뿐가끔씩 차가운 그 앨느낄 때도 있어요하지만 이제는 아무것도요구 할 수 없다는 걸 잘 알죠나 이제 결혼해그 애의 말 듣고한참을 아무 말도할 수가 없었죠그리고 울었죠그 애 마지막 말사랑해 듣고싶던그 한 마디 때문에'
# line = kkma.sentences(sentence)
#
# for l in line:
#     data_test = kkma.morphs(l)
#     print("data_test", data_test)
#
#     for k in range(len(sad_emotion)):
#         for i in emotion_list[k]:
#             for j in range(len(data_test)):
#                 if i[0] == data_test[j]:
#                     print('sad', i[0], i[1])
#                     sad_data_dic[sad_emotion[k]] = sad_data_dic[sad_emotion[k]]+i[1]
#     print(sad_data_dic)
#
#     for k in range(len(happy_emotion)):
#         for i in emotion_list[k + 4]:
#             for j in range(len(data_test)):
#                 if i[0] == data_test[j]:
#                     print('happy', i[0], i[1])
#                     happy_data_dic[happy_emotion[k]] = happy_data_dic[happy_emotion[k]]+i[1]
#     print(happy_data_dic)
#
#     for k in range(len(angry_emotion)):
#         for i in emotion_list[k + 8]:
#             for j in range(len(data_test)):
#                 if i[0] == data_test[j]:
#                     print('angry', i[0], i[1])
#                     angry_data_dic[angry_emotion[k]] = angry_data_dic[angry_emotion[k]]+i[1]
#     print(angry_data_dic)
#
#
# best_emotion = max(sad_data_dic.items(), key=operator.itemgetter(1))[0]
# print("sad result", best_emotion, sad_data_dic[best_emotion])
#
# best_emotion = max(happy_data_dic.items(), key=operator.itemgetter(1))[0]
# print("happy result", best_emotion, happy_data_dic[best_emotion])
#
# best_emotion = max(angry_data_dic.items(), key=operator.itemgetter(1))[0]
# print("angry result", best_emotion, angry_data_dic[best_emotion])
#
#
# # # LSTM 모델 학습을 위한 라벨링 작업
# # result_pd = pd.Series(result_list)
# # data_label = pd.concat([data_temp, result_pd], axis=1)
# # label_df = {"슬픔": 0, "부정": 1, "무관심": 2, "분노": 3}
# # data_label["label"] = data_label.iloc[:, 1].map(label_df)
# # data_label.head(10)
# #
# # class TextLSTM(nn.Module):
# #     def __init__(self):
# #         super(TextLSTM, self).__init__()
# #
# #         self.lstm = nn.LSTM()
#
#
#
#
