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
Run the train_emotion_classification.py file
```
python3 train_emotion_classfication.py
```

### 학습된 모델 기반(.pth 파일을 사용)으로 이미지의 표정 예측
Run the emotion.py file
```
python3 emotion.py
```

