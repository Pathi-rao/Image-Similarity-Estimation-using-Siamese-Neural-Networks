import torch
import torch.nn as nn
import torch.nn.functional as F

""" From: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py """
""" Currently ignoring the 2nd fully copnected layer, which is the trained similarity metric """

class SiameseNetwork(nn.Module):
    def __init__(self, lastLayer = True, pretrained = False):

        super(SiameseNetwork, self).__init__()

        self.lastLayer = lastLayer
        self.pretrained = pretrained
        
        # 3 channel 900x900
        self.conv1 = nn.Conv2d(3, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(16384, 1024)

        if self.lastLayer:
            self.extraL = nn.Linear(1024, 5)

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in')

        #self.net_parameters = []
        #for param in self.conv1.parameters():
        #    param.requires_grad = True
        #    self.net_parameters.append(param)

    def sub_forward(self, x):

        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        # out = torch.sigmoid(self.fc1(out))
        out = self.fc1(out)
        out = self.extraL(out)
        return out

    def forward(self, input1, input2):
        output1 = self.sub_forward(input1)
        output2 = self.sub_forward(input2)

        # if self.lastLayer:
        #     # compute l1 distance (similarity) between the 2 encodings
        #     diff = torch.abs(output1 - output2)
        #     scores = self.extraL(diff)
        #     return scores
        # else:
        return output1, output2

