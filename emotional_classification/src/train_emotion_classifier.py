import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

from emotional_classification.src.utils.dataset import x_train, y_train, x_test, y_test
from emotional_classification.src.utils.dataset import classes

from emotional_classification.src.models.cnn import MiniXception


# Dataset
class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = self.images[index]
        label = self.labels[index]
        sample = (img, label)

        return sample


# Parameters
batch_size = 32
learning_rate = 0.001
n_epoch = 50
input_shape = (3, 48, 48)
num_classes = 5                 # 'angry', 'happy', 'neutral', 'sad', 'surprised'

save_model_pth = "new_model2.pth"
load_model_pth = "new_model2.pth"

train_data = Dataset(images=x_train, labels=y_train)
test_data = Dataset(images=x_test, labels=y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Select model
# model = CNN(num_classes)
model = MiniXception(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Function to save the model
def saveModel():
    torch.save(model.state_dict(), "../trained_models/emotion_models/" + save_model_pth)


# Training
train_losses = []
train_accs = []
test_losses = []
test_accs = []

def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    summary(model, input_shape, device='cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        train_loss = 0.0
        evaluation = []

        for i, data in enumerate(train_loader, 0):
            features, labels = data
            labels = labels.long().to(device)
            features = features.to(device)
            optimizer.zero_grad()

            outputs = model(features.to(torch.float))
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted == labels.cpu()).tolist())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Statistical output
        train_loss = train_loss / (i + 1)
        evaluation = [item for sublist in evaluation for item in sublist]
        train_acc = sum(evaluation) / len(evaluation)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Testing
        if (epoch + 1) % 1 == 0:
            model.eval()
            test_loss = 0.0
            evaluation = []

            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    features, labels = data
                    labels = labels.long().to(device)
                    features = features.to(device)

                    outputs = model(features.to(torch.float))

                    _, predicted = torch.max(outputs.cpu().data, 1)

                    evaluation.append((predicted == labels.cpu()).tolist())
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                test_loss = test_loss / (i + 1)
                evaluation = [item for sublist in evaluation for item in sublist]
                test_acc = sum(evaluation) / len(evaluation)  # accuracy

                test_losses.append(test_loss)
                test_accs.append(test_acc)

                print('[%d, %3d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f' % (
                    epoch + 1, n_epoch, train_loss, train_acc, test_loss, test_acc))


        # Save the model when the accuracy is the best
        if test_acc > best_accuracy:
            saveModel()
            best_accuracy = test_acc


# Function to test what classes performed well
def testClassess():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            features = features.float().to(device)
            outputs = model(features)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Output: Accuracy for each classification
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


if __name__ == "__main__":
    # Let's build our model
    train(n_epoch)
    print('Finished Training')

    # Test which classes performed well
    # testModelAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = MiniXception(num_classes)
    model.load_state_dict(torch.load("../trained_models/emotion_models/" + load_model_pth))

    testClassess()

    # ==================================
    # === 학습/테스트, loss/정확도 시각화 ===
    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(test_losses)), test_losses, label='test loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(train_accs)), train_accs, label='train acc')
    plt.plot(range(len(test_accs)), test_accs, label='test acc')
    plt.legend()
    plt.show()
