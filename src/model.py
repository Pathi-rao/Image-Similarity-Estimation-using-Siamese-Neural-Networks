import torch
import torch.nn as nn
import torch.nn.functional as F

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
        
#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential(
            
#             nn.Conv2d(1, 96, kernel_size=11,stride=1),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),
            
#             nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
#             nn.ReLU(inplace=True),
#             nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),

#             nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2),
#             nn.Dropout2d(p=0.3),

#         )
        
#         # Defining the fully connected layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(30976, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
            
#             nn.Linear(1024, 128),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(128,2))
        
#     def forward_once(self, x):
#         # Forward pass 
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         # forward pass of input 1
#         output1 = self.forward_once(input1)
#         # forward pass of input 2
#         output2 = self.forward_once(input2)
#         return output1, output2






""" From: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py """

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

