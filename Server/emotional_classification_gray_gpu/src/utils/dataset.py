import os

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Absolute path to dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

# Datasets Path
train_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/clean_datasets/train'
test_dir = os.path.abspath(os.path.join(parent_dir, os.pardir)) + '/clean_datasets/test'


batch_size = 32
transforms_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms_train)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms_test)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
