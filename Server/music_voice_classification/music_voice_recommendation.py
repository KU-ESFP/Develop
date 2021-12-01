import os
import csv
import torch
import torchaudio
import operator
import random
import pandas as pd

from konlpy.tag import Kkma
from pydub import AudioSegment
from Server.music_voice_classification.src.models.model import DEVICE
from Server.music_voice_classification.src.models.model import multitemporalfeturemap


# # model
# classes = ['slow', 'fast']
# model = multitemporalfeturemap(2, 2).to(DEVICE)  # channel, classes num

# path name
model_genre_path = 'music_genre_02.pth'
top100_best_genre = 'best_genres_01.csv'
#
# # model load
# model.load_state_dict(torch.load('./trained_models/' + model_genre_path))
# model.eval()
#
# '''
#     top100 songs into 3 secs
# '''
# audio_top100_path = './content/top100/audio_top100'
# for audio_name in os.listdir(audio_top100_path):
#     if os.path.isfile(os.path.join(audio_top100_path, audio_name)):
#         song = os.path.join(audio_top100_path, audio_name)
#
#         newAudio = AudioSegment.from_wav(song)
#         new1 = newAudio[36000:39000]
#         new2 = newAudio[66000:69000]
#         new3 = newAudio[69000:72000]
#         new1.export('./content/top100/audio_top100_3sec/' + audio_name[:-4] + '-1.wav', format="wav")
#         new2.export('./content/top100/audio_top100_3sec/' + audio_name[:-4] + '-2.wav', format="wav")
#         new3.export('./content/top100/audio_top100_3sec/' + audio_name[:-4] + '-3.wav', format="wav")
#
#
# '''
#     prediction the genre of music
# '''
# audio_top100_3sec_path = './content/top100/audio_top100_3sec'
# song_name = []
# song_best_genre = []
# idx = 0
# classes_dic = {'fast': 0, 'slow': 0}
# for audio_name in os.listdir(audio_top100_3sec_path):
#     if os.path.isfile(os.path.join(audio_top100_3sec_path, audio_name)):
#         song = os.path.join(audio_top100_3sec_path, audio_name)
#
#         audio, sr = torchaudio.load(song)                   # [2, 48000]
#         audio = audio.unsqueeze(0)                          # [1, 2, 48000]
#         audio = audio.to(DEVICE)
#
#         output = model(audio)
#         pred = torch.argmax(output, 1)
#
#         if classes[pred] == 'fast':
#             classes_dic['fast'] += 1
#         else:
#             classes_dic['slow'] += 1
#
#         idx += 1
#
#         if idx == 3:    # classified into three per song
#             best_genre = max(classes_dic.items(), key=operator.itemgetter(1))[0]
#             song_name.append(audio_name[:-6])
#             song_best_genre.append(best_genre)
#             print(audio_name, best_genre)
#             print(classes_dic)
#
#             idx = 0
#             classes_dic = {'fast': 0, 'slow': 0}
#
#
# '''
#     save songs genre
# '''
# df = pd.DataFrame({'song': song_name, 'genre': song_best_genre})
# df.to_csv('./content/top100/' + top100_best_genre, index=False, mode='w', encoding='utf-8-sig')
#

'''
    classification using a word dictionary
'''

def saveOutputSong(song_info, emotion):
    output_path = "./content/top100/txt_top100_output"
    f = open(os.path.join(output_path, emotion + '.txt'), 'a', encoding='UTF8')
    f.write(song_info + '\n')
    f.close()


kkma = Kkma()
top100_input_txt_path = './content/top100/txt_top100_input'
fast_genres = 'happy angry'
fast_genres = fast_genres.split()
slow_genres = 'neutral sad'
slow_genres = slow_genres.split()

# top100 정보를 담은 리스트
list_top100 = []


with open('./content/top100/' + top100_best_genre, 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    header = next(reader)       # header omitted

    for song_name, genre in reader:
        list_top100.append(song_name)
        print(song_name + ": ", genre)

        fread = open(os.path.join(top100_input_txt_path, song_name + '.txt'), 'r', encoding='utf-8')   # load song txt
        song = fread.readline()
        line = kkma.sentences(song)

        if genre == 'fast':
            fast_dic = {'happy': 0, 'angry': 0}

            for ll in line:
                print("[" + ll + "]")
                data_morphs = kkma.morphs(ll)

                for ff in fast_genres:
                    is_word = 0
                    words = open(f'./content/w2v_word_dic/{"w2v_" + ff}.txt', 'r', encoding='utf-8')
                    for word_info in words:
                        word, _ = word_info.split(',')

                        for dt in range(len(data_morphs)):
                            if word == data_morphs[dt]:
                                is_word = 1
                                print(ff, word, ": ", ll)
                                break

                        if is_word:
                            break

                    if is_word:
                        fast_dic[ff] += 1
                    words.close()

            fast_best_genre = max(fast_dic.items(), key=operator.itemgetter(1))[0]
            print(song_name + ": ", fast_best_genre)
            saveOutputSong(song_name, fast_best_genre)  # new top100 info
            print(fast_dic)
            print()

        else:
            slow_dic = {'neutral': 0, 'sad': 0}

            for ll in line:
                print("[" + ll + "]")
                data_morphs = kkma.morphs(ll)

                for ff in slow_genres:
                    is_word = 0
                    words = open(f'./content/w2v_word_dic/{"w2v_" + ff}.txt', 'r', encoding='utf-8')
                    for word_info in words:
                        word, _ = word_info.split(',')

                        for dt in range(len(data_morphs)):
                            if word == data_morphs[dt]:
                                is_word = 1
                                print(ff, word, ": ", ll)
                                break

                        if is_word:
                            break

                    if is_word:
                        slow_dic[ff] += 1
                    words.close()

            slow_best_genre = max(slow_dic.items(), key=operator.itemgetter(1))[0]
            print(song_name + ": ", slow_best_genre)
            saveOutputSong(song_name, slow_best_genre)  # new top100 info
            print(slow_dic)
            print()


# 감정이 surprise 일 때는 top1-100 중 18곡 랜덤으로 구해주기
list_surprised = random.sample(list_top100, 18)
for data in list_surprised:
    saveOutputSong(data.rstrip('.txt'), 'surprised')
