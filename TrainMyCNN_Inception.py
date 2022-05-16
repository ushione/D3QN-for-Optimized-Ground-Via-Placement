# -*- coding: utf-8 -*-
# @Python  ：python 3.7
# @Time    : 2021/3/19 16:43
# @Author  : Zheming Gu / 顾哲铭
# @Email   : guzheming@zju.edu.cn
# @File    : CombineNet.py
# @Software: PyCharm
# @Remark  : 比较CNN、带池化和DNN三种网络的性能
# ---------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

seed = 888
if cuda:
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)
print("Ready")

BATCH_SIZE = 32
LEARNING_RATE = 0.0001

train_path = './Data/add.csv'
train_df = pd.read_csv(train_path, header=None)
test_path = './Data/test.csv'
test_df = pd.read_csv(test_path, header=None)

print("shape(train_df):", np.shape(train_df), "\tshape(test_df):", np.shape(test_df))
train_df = np.array(train_df)
test_df = np.array(test_df)

train_data_x, train_data_y, test_data_x, test_data_y = Tensor(train_df[:, :100]), Tensor(
    train_df[:, -1:]), Tensor(
    test_df[:, :100]), Tensor(test_df[:, -1:])
print("shape(train_data_x):", np.shape(train_data_x), "\tshape(test_data_x):", np.shape(test_data_x))
print(np.shape(train_data_x), np.shape(train_data_y), np.shape(test_data_x), np.shape(test_data_y), "\t---------------")

trainDataSet = TensorDataset(train_data_x, train_data_y)
testDataSet = TensorDataset(test_data_x, test_data_y)

train_loader = DataLoader(dataset=trainDataSet, batch_size=BATCH_SIZE, shuffle=True, )


class CNN_Inception(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN_Inception, self).__init__()

        self.branch0 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 1), padding=0),
            nn.ReLU(True),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),
        )

        self.branch2 = nn.Sequential(
            # input shape: batch*1*10*10
            nn.Conv2d(1, 8, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), padding=3),
            nn.ReLU(True),
        )

        self.conv1 = nn.Conv2d(25, 12, kernel_size=(3, 3), padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12 * 10 * 10, 128)
        self.linear2 = nn.Linear(128, 1)

    def _forward(self, x):
        # branch1x1 = self.branch1x1(x)
        # print(branch1x1.shape)

        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = torch.cat((branch0, branch1, branch2, branch3), dim=1)
        # print(np.shape(outputs))
        return outputs

    def forward(self, x):
        x = x.reshape(-1, 1, 10, 10)
        output = self._forward(x)
        output = F.relu(self.conv1(output))
        output = self.flatten(output)
        output = F.relu(self.linear1(output))
        output = self.linear2(output)
        return output


CNN_Inception = CNN_Inception()
CNN_Inception.to(device)

print(CNN_Inception)


optimizer_CNN_Inception = torch.optim.Adam(CNN_Inception.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

r = ['.', '. .', '. . .']

figure, ax = plt.subplots(1, 3, figsize=(15, 4))
plt.ion()

CNN_Normal_Train_Loss, CNN_Inception_Train_Loss, CNN_Normal_Test_Loss, CNN_Inception_Test_Loss = [], [], [], []
CNN_Normal_Train_Acc_1dB, CNN_Inception_Train_Acc_1dB, CNN_Normal_Test_Acc_1dB, CNN_Inception_Test_Acc_1dB = [], [], [], []
CNN_Normal_Train_Acc_3dB, CNN_Inception_Train_Acc_3dB, CNN_Normal_Test_Acc_3dB, CNN_Inception_Test_Acc_3dB = [], [], [], []

Epochs = []
val = 1

for epoch in tqdm(range(1000)):
    if (epoch + 1) % 1000 == 0:
        LEARNING_RATE = LEARNING_RATE / 10
        # optimizer_CNN_Normal = torch.optim.Adam(CNN_Normal.parameters(), lr=LEARNING_RATE)
        optimizer_CNN_Inception = torch.optim.Adam(CNN_Inception.parameters(), lr=LEARNING_RATE)

    CNN_Inception.train()
    for step, (x, y) in enumerate(train_loader):
        # -------------------------------------------------

        prediction_CNN_Inception = CNN_Inception(x)
        loss_CNN_Inception = loss_func(prediction_CNN_Inception, y)

        optimizer_CNN_Inception.zero_grad()
        loss_CNN_Inception.backward()
        optimizer_CNN_Inception.step()
        torch.save(CNN_Inception.state_dict(), 'CNN_Inception.pkl')

        # -------------------------------------------------

    Epochs.append(epoch)

    CNN_Inception.eval()
    # Check Loss and Acc
    with torch.no_grad():
        print("\n,current_lr:", optimizer_CNN_Inception.state_dict()['param_groups'][0]['lr'])

        Output_CNN_Inception_Train = CNN_Inception(train_data_x)
        Output_CNN_Inception_Train_Loss = loss_func(Output_CNN_Inception_Train, train_data_y)
        CNN_Inception_Train_Loss.append(Output_CNN_Inception_Train_Loss)
        Output_CNN_Inception_Train_Acc_1dB = abs(Output_CNN_Inception_Train - train_data_y) <= 1
        CNN_Inception_Train_Acc_1dB.append(np.count_nonzero(Output_CNN_Inception_Train_Acc_1dB.cpu()) / 40)
        Output_CNN_Inception_Train_Acc_3dB = abs(Output_CNN_Inception_Train - train_data_y) <= 3
        CNN_Inception_Train_Acc_3dB.append(np.count_nonzero(Output_CNN_Inception_Train_Acc_3dB.cpu()) / 40)
        print("Epoch:", epoch, "Output_CNN_Inception_Train_Loss：", Output_CNN_Inception_Train_Loss,
              "CNN_Inception_Train_Acc_1dB:", CNN_Inception_Train_Acc_1dB[-1], "%",
              "CNN_Inception_Train_Acc_3dB:", CNN_Inception_Train_Acc_3dB[-1], "%")

        Output_CNN_Inception_Test = CNN_Inception(test_data_x)
        Output_CNN_Inception_Test_Loss = loss_func(Output_CNN_Inception_Test, test_data_y)
        CNN_Inception_Test_Loss.append(Output_CNN_Inception_Test_Loss)
        Output_CNN_Inception_Test_Acc_1dB = abs(Output_CNN_Inception_Test - test_data_y) <= 1
        CNN_Inception_Test_Acc_1dB.append(np.count_nonzero(Output_CNN_Inception_Test_Acc_1dB.cpu()) / 10)
        Output_CNN_Inception_Test_Acc_3dB = abs(Output_CNN_Inception_Test - test_data_y) <= 3
        CNN_Inception_Test_Acc_3dB.append(np.count_nonzero(Output_CNN_Inception_Test_Acc_3dB.cpu()) / 10)
        print("Epoch:", epoch, "Output_CNN_Inception_Test_Loss:", Output_CNN_Inception_Test_Loss,
              "CNN_Inception_Test_Acc_1dB:", CNN_Inception_Test_Acc_1dB[-1], "%",
              "CNN_Inception_Test_Acc_3dB:", CNN_Inception_Test_Acc_3dB[-1], "%")

    ax[0].cla()
    ax[0].plot(Epochs, CNN_Inception_Train_Loss, color='pink', label='CNN_Inception train loss')
    ax[0].plot(Epochs, CNN_Inception_Test_Loss, color='red', label='CNN_Inception test loss')
    ax[0].set_title('Loss vs. Epochs')
    # ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].cla()
    ax[1].plot(Epochs, CNN_Inception_Train_Acc_1dB, color='pink', label='CNN_Inception train accuracy')
    ax[1].plot(Epochs, CNN_Inception_Test_Acc_1dB, color='red', label='CNN_Inception test accuracy')
    ax[1].set_title('Accuracy(1dB) vs. Epochs')
    ax[1].legend()

    ax[2].cla()
    ax[2].plot(Epochs, CNN_Inception_Train_Acc_3dB, color='pink', label='CNN_Inception train accuracy')
    ax[2].plot(Epochs, CNN_Inception_Test_Acc_3dB, color='red', label='CNN_Inception test accuracy')
    ax[2].set_title('Accuracy(3dB) vs. Epochs')
    ax[2].legend()

    plt.pause(0.1)
    plt.savefig('CNN_Inception_Performance.jpg')

    writeResult = np.array(
        [Epochs, CNN_Inception_Train_Loss, CNN_Inception_Test_Loss,
         CNN_Inception_Train_Acc_1dB, CNN_Inception_Test_Acc_1dB,
         CNN_Inception_Train_Acc_3dB, CNN_Inception_Test_Acc_3dB]).T
    writeResult_df = pd.DataFrame(writeResult)
    writeResult_df.to_csv("CNN_Inception_Performance.csv")

plt.ioff()
plt.show()
