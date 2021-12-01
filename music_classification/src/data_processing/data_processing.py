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
# happy_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_happy_03.model')
angry_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_angry_01.model')

# sad_emotion = ['사랑', '슬픔', '분노', '무관심']
# happy_emotion = ['사랑', '슬픔', '긍정', '행복']

# sad_emotion = ['이별', '사랑', '슬픔', '눈물', '분노', '무관심']
# happy_emotion = ['사랑', '행복', '예쁘', '미소', '자유']

# sad_emotion = ['슬픔', '후회', '그리움', '외로움', '무관심', '분노']
# happy_emotion = ['연애', '기대', '자신감', '긍정', '기쁨', '행복', '감동', '사랑']
#angry_emotion = ['실망', '부시', '밟', '끔찍', '시발', '새끼', '짜증', '두렵', '분노']
angry_emotion = ['실망']

w2v_emotion_path = 'w2v_emotion_words'


print('angry')
for index in range(len(angry_emotion)):
    emotion_one = []
    print(angry_emotion[index])
    fw = open(w2v_emotion_path + '/w2v_angry_01.txt', 'a', encoding='utf8')
    for word in angry_model.wv.most_similar(angry_emotion[index], topn=2000):
        print(word)
        if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
            emotion_one.append(word)
            fw.write(word[0] + ', ' + str(word[1]) + '\n')
    if emotion_one:
        emotion_list.append(emotion_one)
    print()
print()
fw.close()


print()
print("emotion_list", emotion_list)
print('총 테스트의 개수: {}'.format(len(emotion_list)))


# ==================================================================================

angry_song_path = '../../song_datasets/angry_songs'
neutral_song_path = '../../song_datasets/neutral_songs'

angry_word_data = 'w2v_angry.txt'
neutral_word_data = 'w2v_neutral.txt'

song_save_path = '../../csv_datasets/'
angry_datasets = 'angry_datasets_01.csv'
neutral_datasets = 'neutral_datasets_01.csv'

emotion = int(input('emotion number [angry:0, neutral:1]: '))   # 0이면 angry, 1이면 neutral
if emotion == 0:
    angry_sentences = []
    for song_name in os.listdir(angry_song_path):
        if os.path.isfile(os.path.join(angry_song_path, song_name)):
            fread = open(os.path.join(angry_song_path, song_name), 'r', encoding='utf-8')
            song = fread.readline()
            line = kkma.sentences(song)
            for ll in line:
                print(song_name, ll)
                is_angry = 0
                data_test = kkma.morphs(ll)

                angry_words = open('./w2v_emotion_words/' + angry_word_data, 'r', encoding='utf-8')
                for h_word in angry_words:
                    sw = h_word.split(',')
                    for dt in range(len(data_test)):
                        if sw[0] == data_test[dt]:
                            is_angry = 1
                            print('[angry]', sw[0], ": ", ll)
                            break
                    if is_angry:
                        break

                if is_angry:
                    angry_sentences.append(ll)
                angry_words.close()


    df = pd.DataFrame({'text': angry_sentences, 'emotion': 0})  # angry
    if not os.path.exists(song_save_path + angry_datasets):
        df.to_csv(song_save_path + angry_datasets, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(song_save_path + angry_datasets, index=False, mode='a', encoding='utf-8-sig', header=False)

else:
    neutral_sentences = []
    for song_name in os.listdir(neutral_song_path):
        if os.path.isfile(os.path.join(neutral_song_path, song_name)):
            fread = open(os.path.join(neutral_song_path, song_name), 'r', encoding='utf-8')
            song = fread.readline()
            line = kkma.sentences(song)
            for ll in line:
                print(song_name, ll)
                is_neutral = 0
                data_test = kkma.morphs(ll)

                neutral_words = open('./w2v_emotion_words/' + neutral_word_data, 'r', encoding='utf-8')
                for s_word in neutral_words:
                    sw = s_word.split(',')
                    for dt in range(len(data_test)):
                        if sw[0] == data_test[dt]:
                            is_neutral = 1
                            print('[neutral]', sw[0], ": ", ll)
                            break
                    if is_neutral:
                        break

                if is_neutral:
                    neutral_sentences.append(ll)
                neutral_words.close()

    df = pd.DataFrame({'text': neutral_sentences, 'emotion': 2})  # neutral
    if not os.path.exists(song_save_path + neutral_datasets):
        df.to_csv(song_save_path + neutral_datasets, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(song_save_path + neutral_datasets, index=False, mode='a', encoding='utf-8-sig', header=False)

