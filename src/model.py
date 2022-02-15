import torch
import torch.nn as nn
import torch.nn.functional as F


""" From: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py """

class SiameseNetwork(nn.Module):
    def __init__(self, lastLayer=True):
        self.lastLayer = lastLayer

        super(SiameseNetwork, self).__init__()
        
        # Koch et al.
        # Conv2d(input_channels, output_channels, kernel_size) 
        #Input should be 105x105x1
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        # self.fc2 = nn.Linear(4096, 1)

        # 3 channels 150x150
        #self.conv1 = nn.Conv2d(3, 64, 10)
        #self.conv2 = nn.Conv2d(64, 128, 7)
        #self.conv3 = nn.Conv2d(128, 128, 4)
        #self.conv4 = nn.Conv2d(128, 256, 4)
        #self.fc1 = nn.Linear(30976, 4096)
        #self.fc2 = nn.Linear(4096, 1)

        # 1 channel 200x200
        # self.conv1 = nn.Conv2d(1, 64, 10)
        # self.conv2 = nn.Conv2d(64, 128, 7)
        # self.conv3 = nn.Conv2d(128, 128, 4)
        # self.conv4 = nn.Conv2d(128, 256, 4)
        # self.fc1 = nn.Linear(30976, 4096)

        if self.lastLayer:
            self.extraL = nn.Linear(4096, 1)
        
        # using kaiming intialization
        # random source- https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

        # distanceLayer = True  
        # defines if the last layer uses a distance metric or a neuron output
        
    def forward_once(self, x):
        # out = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        # out = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        # out = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(out))), 2)
        # out = self.conv4_bn(F.relu(self.conv4(out)))

        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        out = torch.sigmoid(self.fc1(out))
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.lastLayer:
            # compute l1 distance (similarity) between the 2 encodings
            diff = torch.abs(output1 - output2)
            scores = self.extraL(diff)
            return scores
        else:
            return output1, output2


########### +++++++++++++++ ###########

#create the Siamese Neural Network
# class SiameseNetwork(nn.Module):

#     def __init__(self):
#         super(SiameseNetwork, self).__init__()

#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11,stride=4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, stride=2),
            
#             nn.Conv2d(96, 256, kernel_size=5, stride=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 384, kernel_size=3,stride=1),
#             nn.ReLU(inplace=True)
#         )

#         # Setting up the Fully Connected Layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(384, 1024),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(1024, 256),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(256,2)
#         )
        
#     def forward_once(self, x):
#         # This function will be called for both images
#         # Its output is used to determine the similiarity
#         output = self.cnn1(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         # In this function we pass in both images and obtain both vectors
#         # which are returned
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)

#         return output1, output2