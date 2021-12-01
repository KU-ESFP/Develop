# Android Programming (JAVA)

## Packages
- **activity**
  - BaseActivity
  - LauncherActivity
  - SurveyActivity
  - MainActivity
  - AnalaysisActivity
  - ResultActivity
- **utils**
  - BFunction
  - Const
  - NetUtil
  - PopupUtil
  - ToastUtil
- **network**
  - NetworkManager
  - ResEmotion

## About
- Android Stuido 4.1.1
- SQLite
- dependency
  - glide 4.12.0
  - OkHttp3 4.9.0
  - Retrofit2 2.3.0
  - gson 2.8.0


---

# Emotional Classification
얼굴 표정 검출에 관한 딥러닝 기술

## Setting 
* Python 3.7 (py3.7)
* opencv 3.4.2.16
* PyCharm + Anaconda3

## Module
* Anaconda Prompt
  * conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
  * pip install
    * numpy==1.19.3
    * matplotlib
    * scikit-image
    * scipy
    * opencv-python==3.4.2.16
    * opencv-contrib-python==3.4.2.16
    * sklearn
    * torchsummary

## Instructions
### 감정 분류를 위한 모델 학습 방법
Run the train_emotion_classifier.py file
```
python3 train_emotion_classifier.py
```

### 학습된 모델 기반(.pth 파일을 사용)으로 이미지의 표정 예측
Run the emotion.py file
```
python3 emotion.py
```

---

# Music Voice Classification
감정에 따른 음악 추천 딥러닝 기술  

### 설명
>
> - 음악 속도에 따라 음원을 두 부류로 나눔 (fast or slow)
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
### 음원 속도를 기반으로 fast, slow 분류 모델 학습
Run the train_music_voice_classifier.py file
```
python3 train_music_voice_classifier.py
```
  

### 학습된 모델 기반(.pth 파일을 사용)으로 장르 분류 및 음악 추천
Run the music_voice_recommendation file
```
python3 music_voice_recommendation.py
```


