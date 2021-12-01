import os
import torch
import torch.nn as nn

import time
from torch import optim
from torch.optim import Adam
from torchsummary import summary

from Server.emotional_classification_gray_gpu.src.models.model import CNN7
from Server.emotional_classification_gray_gpu.src.utils.dataset import train_loader, test_loader
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# parameters
batch_size = 32
learning_rate = 0.001
n_epoch = 60
input_shape = (1, 48, 48)
num_classes = 5

classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']

save_model_pth = "new_model_CNN7_result.pth"
load_acc_model_pth = "acc_new_model_CNN7_result.pth"
load_loss_model_pth = "loss_new_model_CNN7_result.pth"


# Instantiate a neural network model
model = CNN7(num_classes)


# 학습 데이터에 대한 모델 학습
train_losses = []
train_accs = []
test_losses = []
test_accs = []


# Function to save the model
def saveModel(type):
    torch.save(model.state_dict(), "../trained_models/emotion_models/" + type + "_" + save_model_pth)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0
    best_val_loss = 1.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    summary(model, input_shape, device='cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        train_loss = 0.0
        evaluation = []
        model.train()                                                           # 모델을 학습 모드로 전환

        for i, data in enumerate(train_loader):
            features, labels = data
            labels = labels.to(device)
            features = features.to(device)
            optimizer.zero_grad()                                               # 변화도(Gradient) 매개변수를 0으로 만듦

            # 순전파 + 역전파 + 최적화
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)                     # 출력의 제일 큰 값의 index 반환
            evaluation.append((predicted == labels).tolist())             # 정답과 비교하여 true, false 값을 저장

            loss = criterion(outputs, labels)                                   # 출력과 라벨을 비교하여 loss 계산
            loss.backward()                                                     # 역전파, 기울기 계산
            optimizer.step()                                                    # 가중치 값 업데이트, 학습 한번 진행

            train_loss += loss.item()                                           # loss를 train_loss에 누적
            # sched.step()

        # 통계 출력
        train_loss = train_loss / (i + 1)                                       # 평균 train_loss 구하기
        evaluation = [item for sublist in evaluation for item in sublist]       # [true, false] 값을 list로 저장
        train_acc = sum(evaluation) / len(evaluation)                           # true 비율 계산

        train_losses.append(train_loss)                                         # 해당 epoch의 train loss 기록
        train_accs.append(train_acc)                                            # 해당 epoch의 train acc 기록

        # ============== 테스트 =============
        if (epoch + 1) % 1 == 0:
            test_loss = 0.0                                                     # test loss 초기화
            evaluation = []  # accuracy                                         # 예측 정확 여부 저장할 list

            with torch.no_grad():
                model.eval()  # 모델을 평가모드로 전환
                for i, data in enumerate(test_loader):                          # 각 batch 마다
                    features, labels = data                                     # 데이터 특징과 라벨로 나누기
                    labels = labels.to(device)
                    features = features.to(device)

                    outputs = model(features)

                    _, predicted = torch.max(outputs, 1)

                    evaluation.append((predicted == labels).tolist())
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                test_loss = test_loss / (i + 1)
                evaluation = [item for sublist in evaluation for item in sublist]
                test_acc = sum(evaluation) / len(evaluation)  # accuracy

                test_losses.append(test_loss)
                test_accs.append(test_acc)

                print('Epoch[%d], loss: %.4f, Accuracy : %.4f, val-loss: %.4f, val_acc : %.4f' % (
                    epoch, train_loss, train_acc, test_loss, test_acc))

        # 정확도가 가장 높은 최적의 모델 저장
        if test_acc > best_accuracy:
            saveModel('acc')
            best_accuracy = test_acc

        # 검증 오차가 가장 적은 최적의 모델을 저장
        if test_loss < best_val_loss:
            saveModel('loss')
            best_val_loss = test_loss


def classesTest():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 각 분류(class)에 대한 예측값 계산을 위해 준비
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # 변화도는 여전히 필요하지 않음
    with torch.no_grad():
        model.eval()  # 모델을 평가모드로 전환
        for i, data in enumerate(test_loader):  # 각 batch 마다
            features, labels = data  # 데이터 특징과 라벨로 나누기
            labels = labels.to(device)
            features = features.to(device)

            outputs = model(features)

            _, predicted = torch.max(outputs, 1)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # 각 분류별 정확도(accuracy)를 출력합니다
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


if __name__ == "__main__":
    # Let's build our model
    train(n_epoch)
    print('Finished Training')

    model = CNN7(num_classes)

    print("=======================================================")
    model.load_state_dict(torch.load("../trained_models/emotion_models/" + load_acc_model_pth))
    classesTest()

    print("=======================================================")
    model.load_state_dict(torch.load("../trained_models/emotion_models/" + load_loss_model_pth))
    classesTest()