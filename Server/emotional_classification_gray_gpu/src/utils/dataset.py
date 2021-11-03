import os

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Absolute path to dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(os.path.abspath(os.path.join(parent_dir, os.pardir)))

# Datasets Path
train_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/clean_datasets/train'
test_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/clean_datasets/test'


batch_size = 32
transforms_train = transforms.Compose([
    transforms.Grayscale(),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms_train)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms_test)

# batch_size(int) : 반복문을 돌때, 한번에 몇 개의 이미지를 꺼내올지 정함
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# for i in range(1, 17):
#     plt.subplot(4, 4, i)
#     plt.imshow(train_data[0][0].squeeze(), cmap='gray')  # (batch_size, height, width)
#     plt.axis('off')
# plt.show()
#
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(classes[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
#
# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels[0]}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
