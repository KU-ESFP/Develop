import os
import numpy as np
import pandas as pd

from konlpy.tag import Kkma
from sklearn.model_selection import train_test_split


kkma = Kkma()

# path name
data_path = '../../csv_datasets/'
total_dataset = 'song_datasets_v2.csv'
train_dataset = 'train_data_v2.csv'
test_dataset = 'test_data_v2.csv'

''' 
    [데이터셋 정보]
    각 노래의 한 문장에 대한 감정을 라벨링
    - angry: 0, happy: 1, neutral: 2, sad: 3
'''
angry_csv = pd.read_csv('../../csv_datasets/angry_datasets_01.csv')
happy_csv = pd.read_csv('../../csv_datasets/happy_datasets_01.csv')
neutral_csv = pd.read_csv('../../csv_datasets/neutral_datasets_01.csv')
sad_csv = pd.read_csv('../../csv_datasets/sad_datasets_01.csv')

new_df = pd.concat([angry_csv, happy_csv, neutral_csv, sad_csv])  # 감정별 csv 파일 합친 부분
print(new_df)

df = pd.DataFrame(new_df)
if not os.path.exists(data_path + total_dataset):
    df.to_csv(data_path + total_dataset, index=False, mode='w', encoding='utf-8-sig')
else:
    df.to_csv(data_path + total_dataset, index=False, mode='a', encoding='utf-8-sig', header=False)



'''
    [데이터 전처리 01]
    중복 제거 및 NULL 값이 존재하는 행 제거

    [훈련 데이터와 테스트 데이터는 분리]
    데이터를 8:2 비율로 훈련 데이터와 테스트 데이터 분류
    ./csv_datasets/train_data.csv   : 모델 학습할 때 사용할 train data
    ./csv_datasets/test_data.csv    : 모델 검증할 때 사용할 music_test data
'''
dataset = pd.read_csv(data_path + total_dataset, encoding='utf-8')
dataset['text'].unique(), dataset['emotion'].unique()
dataset.drop_duplicates(subset=['text'], inplace=True)
print(dataset.groupby('emotion').size().reset_index(name='count'))

train_data = dataset.dropna(how='any')                          # Null 값이 존재하는 행 제거
print('\nis Null: {}'.format(dataset.isnull().values.any()))    # Null 값이 있는지 다시 확인
print(dataset.isnull().sum())                                   # Null 값이 있다면 어떤 열에 존재하는지 확인
print('dataset 총 개수: {}'.format(len(dataset)))

# 전체 데이터를 훈련 데이터와 테스트 데이터를 8:2 비율로 나누기
train_data, test_data = train_test_split(dataset, test_size=0.1)

train_data.to_csv(data_path + train_dataset, index=False)
test_data.to_csv(data_path + test_dataset, index=False)
