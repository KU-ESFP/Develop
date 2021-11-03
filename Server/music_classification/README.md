# Music Classification
감정에 따른 음악 추천 딥러닝 기술

## Setting 
* Python 3.7 (py3.7)
* opencv 3.4.2.16
* PyCharm + Anaconda3

## Module
* pip install
  * konlpy
  * selenium
  
### Selenium & Webdriver
```
https://dev-pengun.tistory.com/entry/%ED%81%AC%EB%A1%A4%EB%A7%81-python-webdriver%EB%A1%9C-%ED%81%AC%EB%A1%AC%EC%B0%BD-%EC%97%B4%EA%B8%B0-selenium
https://nanchachaa.tistory.com/18           # ERROR 생겼을 때 참조
```

## Instructions
### 감정에 따른 음악 추천 모델 학습 방법
Run the train_music_classifier.py file
```
python3 train_music_classifier.py
```

### 학습된 모델 기반(.pth 파일을 사용)으로 Melon TOP100 에 대한 노래 감정 분류
Run the music_recommendation file
```
python3 music_recommendation.py
```

