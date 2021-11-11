# Music Voice Classification
감정에 따른 음악 추천 딥러닝 기술  

### 설명
>
> - 장르에 따라 음원을 두 부류로 나눔 (fast or slow)
> - Word2Vec을 이용해 만들어진 단어 사전을 이용해 아래와 같이 네 가지로 분류
>    - fast: happy, angry
>    - slow: neutral, sad
>    - music_classification/src/data_processing

## Setting 
* Python 3.7 (py3.7)
* opencv 3.4.2.16
* PyCharm + Anaconda3

## Module
* pip install
  * librosa 
  * xgboost
  * seaborn
  * pydub
  * ffmpeg
  * torchaudio
    

## Instructions
### 음원을 기반으로 장르 분류 모델 학습
Run the train_music_voice_classifier.py file
```
python3 train_music_voice_classifier.py
```
  

### 학습된 모델 기반(.pth 파일을 사용)으로 장르 분류 및 음악 추천
Run the music_voice_recommendation file
```
python3 music_voice_recommendation.py
```

