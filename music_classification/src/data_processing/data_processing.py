# -*- coding: utf-8 -*-
import operator
import os
import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Kkma

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

kkma = Kkma()

emotion_list = []                 # 감정 사전: 감정 단어가 담길 리스트 [모델의 input]

# ==================================================================================
# 감정별 단어 사전 만들기
# ==================================================================================

sad_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_sad_11.model')
happy_model = Word2Vec.load('../../trained_models/word2vec_models/w2v_happy_03.model')

happy_emotion = ['연애', '기대', '자신감', '긍정', '기쁨', '행복', '감동', '사랑', '빛나', '미소', '자유', '씰룩']
sad_emotion = ['사랑', '슬픔', '후회', '무관심', '분노', '그리움', '외로움', '아픔']
sad_emotion = ['울음']

w2v_emotion_path = 'w2v_emotion_words'

print('sad')
for index in range(len(sad_emotion)):
    emotion_one = []
    print(sad_emotion[index])
    fw = open(w2v_emotion_path + '/w2v_sad_04.txt', 'a', encoding='utf8')
    for word in sad_model.wv.most_similar(sad_emotion[index], topn=1500):
        print(word)
        if word[1] >= 0.5:        # 튜플 형식, 1번째 인덱스의 유사도와 비교
            emotion_one.append(word)
            fw.write(word[0] + ', ' + str(word[1]) + '\n')
    if emotion_one:
        emotion_list.append(emotion_one)
    print()
print()
fw.close()


print('happy')
for index in range(len(happy_emotion)):
    emotion_one = []
    print(happy_emotion[index])
    fw = open(w2v_emotion_path + '/w2v_happy_01.txt', 'a', encoding='utf8')
    for word in happy_model.wv.most_similar(happy_emotion[index], topn=2000):
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
# 감정별 csv file 만들기
# ==================================================================================

# sad_song_path = '../../song_datasets/sad_songs'
happy_song_path = '../../song_datasets/happy_songs'
angry_song_path = '../../song_datasets/angry_songs'

# sad_word_data = 'w2v_sad.txt'
happy_word_data = 'w2v_happy.txt'
angry_word_data = 'w2v_angry.txt'

song_save_path = '../../csv_datasets/'
# sad_datasets = 'sad_datasets_01.csv'
happy_datasets = 'happy_datasets_01.csv'

emotion = int(input('emotion number [happy:0, sad:1]: '))   # 0이면 happy, 1이면 sad
if emotion == 0:
    happy_sentences = []
    for song_name in os.listdir(happy_song_path):
        if os.path.isfile(os.path.join(happy_song_path, song_name)):
            fread = open(os.path.join(happy_song_path, song_name), 'r', encoding='utf-8')
            song = fread.readline()
            line = kkma.sentences(song)
            for ll in line:
                print(song_name, ll)
                is_happy = 0
                data_test = kkma.morphs(ll)

                happy_words = open('./w2v_emotion_words/' + happy_word_data, 'r', encoding='utf-8')
                for h_word in happy_words:
                    sw = h_word.split(',')
                    for dt in range(len(data_test)):
                        if sw[0] == data_test[dt]:
                            is_happy = 1
                            print('[happy]', sw[0], ": ", ll)
                            break
                    if is_happy:
                        break

                if is_happy:
                    happy_sentences.append(ll)
                happy_words.close()

    df = pd.DataFrame({'text': happy_sentences, 'emotion': 1})  # happy
    if not os.path.exists(song_save_path + happy_datasets):
        df.to_csv(song_save_path + happy_datasets, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(song_save_path + happy_datasets, index=False, mode='a', encoding='utf-8-sig', header=False)

else:
    angry_sentences = []
    for song_name in os.listdir(angry_song_path):
        if os.path.isfile(os.path.join(angry_song_path, song_name)):
            fread = open(os.path.join(angry_song_path, song_name), 'r', encoding='utf-8')
            song = fread.readline()
            line = kkma.sentences(song)
            for ll in line:
                print("[" + song_name + "]", ll)
                is_angry = 0
                data_test = kkma.morphs(ll)
                # print("[data]", data_test)
                angry_words = open('./w2v_emotion_words/' + angry_word_data, 'r', encoding='utf-8')
                for s_word in angry_words:
                    sw = s_word.split(',')
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






