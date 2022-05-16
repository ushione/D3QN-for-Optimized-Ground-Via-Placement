import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class DNN_Relu(torch.nn.Module):
    def __init__(self):
        super(DNN_Relu, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DNN_Tanh(torch.nn.Module):
    def __init__(self):
        super(DNN_Tanh, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 25),
            torch.nn.Tanh(),
            torch.nn.Linear(25, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class CNN_Inception(nn.Module):
    def __init__(self):
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
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = torch.cat((branch0, branch1, branch2, branch3), dim=1)
        return outputs

    def forward(self, x):
        x = x.reshape(-1, 1, 10, 10)
        output = self._forward(x)
        output = F.relu(self.conv1(output))
        output = self.flatten(output)
        output = F.relu(self.linear1(output))
        output = self.linear2(output)
        return output


def Build_Evaluate_Network(name):
    model = None
    if name == "DNN_Relu":
        model = DNN_Relu()
        model.load_state_dict(torch.load('./Params/DNN_Relu.pkl'))
    elif name == "CNN_Inception":
        model = CNN_Inception()
        model.load_state_dict(torch.load('./Params/CNN_Inception.pkl'))
    else:
        raise Exception("Please choose your evaluation model correctly! The type of model you are trying to build is {}!".format(name))
    return model


if __name__ == '__main__':
    print(Build_Evaluate_Network("DNN_Relu"))
    print(Build_Evaluate_Network("CNN_Inception"))
