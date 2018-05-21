#!/usr/bin/env Python
# coding=utf-8
import os
from data import MyDataset

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as Models
from torch.autograd import Variable


class LOL(nn.Module):
    def __init__(self):
        super(LOL, self).__init__()
        self.feature = Models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.pool = nn.MaxPool2d(kernel_size=(4, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.size()[0]
        x = self.feature(x.view(N * 4, 3, 448, 448))
        x = self.pool(x.view(N, 1, 4, 2048))
        x = nn.functional.normalize(x.view(-1, 2048))
        y = self.fc(x.view(-1, 2048))
        assert y.size() == (N, 1)
        return x.view(-1, 2048), y


def main():
    print("start test!!")
    cur_model = LOL()
    cur_model.load_state_dict(torch.load('./params.pkl'))
    cur_model.eval()
    correct = 0
    test_data = MyDataset()
    test_loader = Data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=6)
    print("test data size: ", len(test_data))
    test_result = []
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x.view(-1, 4, 3, 448, 448), volatile=True)
        b_y = Variable(torch.FloatTensor(y.numpy()).view(-1, 1))

        _, pred = cur_model(b_x)

        for j in range(b_x.size(0)):
            test_result.append([pred.cpu().data[j][0], b_y.cpu().data[j][0]])
        pred[pred > 0.497] = 1
        pred[pred <= 0.497] = 0
        correct += torch.sum(pred == b_y).data[0]

    accuracy = correct * 1.0 / len(test_data)
    print("accuracy: %.4f" % accuracy)

    with open("./prediction.txt", 'w') as f:
        for j in range(len(test_result)):
            f.writelines(str(test_result[j][0]) + " " + str(test_result[j][1]) + "\n")

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()