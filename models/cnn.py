import os.path
from os import path
import torch
import torchvision.datasets as trds
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torchvision.transforms as transforms
from torch.autograd import Variable


class Cnn(Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def train(self, epoch):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        data_path = './data'
        download = not(path.exists(data_path))
        train_set = trds.CIFAR10(root=data_path, train=True,
                                                download=download, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                                  shuffle=True, num_workers=2)

        test_set = trds.CIFAR10(root=data_path, train=False,
                                               download=download, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                         shuffle=False, num_workers=2)
        tr_loss = 0
        for i, data in enumerate(train_loader, 0):
              inputs, labels = data
              x_train, y_train = Variable(inputs), Variable(labels)

              #print(x_train.shape)
              #print(y_train.shape)
        for i, data in enumerate(test_loader, 0):
              inputs, labels = data
              x_test, y_test = Variable(inputs), Variable(labels)

              #print(x_test.shape)
              #print(y_test.shape)

        # prediction for training and validation set
        output_train = self(x_train)
        output_test = model(x_test)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_test = criterion(output_test, y_test)
        train_losses.append(loss_train)
        test_losses.append(loss_test)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        tr_loss = loss_train.item()
        if epoch%2 == 0:
            # printing the validation loss
            print('Epoch : ',epoch+1, '\t', 'test loss :', loss_test)
