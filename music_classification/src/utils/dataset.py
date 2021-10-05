import os
import pandas as pd

from konlpy.tag import Kkma
from sklearn.model_selection import train_test_split


kkma = Kkma()

# path name
save_path = '../../csv_datasets/'
total_dataset = 'song_datasets.csv'
train_dataset = 'train_data.csv'
test_dataset = 'test_data.csv'

''' 
    [데이터셋 정보]
    각 노래의 한 문장에 대한 감정을 라벨링
    - angry: 0, happy: 1, neutral: 2, sad: 3
'''
song_path = '../../song_datasets/angry_songs'
for song_name in os.listdir(song_path):
    if os.path.isfile(os.path.join(song_path, song_name)):
        fread = open(os.path.join(song_path, song_name), 'r', encoding='UTF8')
        song = fread.readline()
        line = kkma.sentences(song)
        df = pd.DataFrame({'text': line, 'emotion': 0})    # angry

        if not os.path.exists(save_path + total_dataset):
            df.to_csv(save_path + total_dataset, index=False, mode='w', encoding='utf-8-sig')
        else:
            df.to_csv(save_path + total_dataset, index=False, mode='a', encoding='utf-8-sig', header=False)

song_path = '../../song_datasets/happy_songs'
for song_name in os.listdir(song_path):
    if os.path.isfile(os.path.join(song_path, song_name)):
        fread = open(os.path.join(song_path, song_name), 'r', encoding='UTF8')
        song = fread.readline()
        line = kkma.sentences(song)
        df = pd.DataFrame({'text': line, 'emotion': 1})    # happy
        df.to_csv(save_path + total_dataset, index=False, mode='a', encoding='utf-8-sig', header=False)

song_path = '../../song_datasets/neutral_songs'
for song_name in os.listdir(song_path):
    if os.path.isfile(os.path.join(song_path, song_name)):
        fread = open(os.path.join(song_path, song_name), 'r', encoding='UTF8')
        song = fread.readline()
        line = kkma.sentences(song)
        df = pd.DataFrame({'text': line, 'emotion': 2})    # neutral
        df.to_csv(save_path + total_dataset, index=False, mode='a', encoding='utf-8-sig', header=False)

song_path = '../../song_datasets/sad_songs'
for song_name in os.listdir(song_path):
    if os.path.isfile(os.path.join(song_path, song_name)):
        fread = open(os.path.join(song_path, song_name), 'r', encoding='UTF8')
        song = fread.readline()
        line = kkma.sentences(song)
        df = pd.DataFrame({'text': line, 'emotion': 3})    # sad
        df.to_csv(save_path + total_dataset, index=False, mode='a', encoding='utf-8-sig', header=False)


# # 감정 별로 몇 개의 노래가 들어있는지 확인
# path = '../../song_datasets/angry_songs'
# file_list = os.listdir(path)
# print(len(file_list))


'''
    [데이터 전처리 01]
    중복 제거 및 NULL 값이 존재하는 행 제거
    
    [훈련 데이터와 테스트 데이터는 분리]
    데이터를 8:2 비율로 훈련 데이터와 테스트 데이터 분류
    ./csv_datasets/train_data.csv   : 모델 학습할 때 사용할 train data 
    ./csv_datasets/test_data.csv    : 모델 검증할 때 사용할 test data
'''
dataset = pd.read_csv(save_path + total_dataset, encoding='utf-8')
dataset['text'].unique(), dataset['emotion'].unique()
dataset.drop_duplicates(subset=['text'], inplace=True)
print(dataset.groupby('emotion').size().reset_index(name='count'))

train_data = dataset.dropna(how='any')                                # Null 값이 존재하는 행 제거
print('\nis Null: {}'.format(dataset.isnull().values.any()))          # Null 값이 있는지 다시 확인
print(dataset.isnull().sum())                                         # Null 값이 있다면 어떤 열에 존재하는지 확인
print('dataset 총 개수: {}'.format(len(dataset)))

# 전체 데이터를 훈련 데이터와 테스트 데이터를 8:2 비율로 나누기
train_data, test_data = train_test_split(dataset, test_size=0.2)

train_data.to_csv(save_path + train_dataset, index=False)
test_data.to_csv(save_path + test_dataset, index=False)
