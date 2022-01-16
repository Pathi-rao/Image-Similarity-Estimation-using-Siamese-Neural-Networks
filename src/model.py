import torch
import torch.nn as nn
import torch.nn.functional as F


""" From: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py """

# class SiameseNetwork(nn.Module):
#     def __init__(self):

#         super(SiameseNetwork, self).__init__()
        
#         # Koch et al.
#         # Conv2d(input_channels, output_channels, kernel_size) 
#         self.conv1 = nn.Conv2d(1, 64, 10)
#         self.conv2 = nn.Conv2d(64, 128, 7)
#         self.conv3 = nn.Conv2d(128, 128, 4)
#         self.conv4 = nn.Conv2d(128, 256, 4)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.dropout1 = nn.Dropout(0.1)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(256 * 6 * 6, 4096)
#         self.fcOut = nn.Linear(4096, 1)
#         self.sigmoid = nn.Sigmoid()


#     def convs(self, x):

#         # Koch et al.
#         # out_dim = in_dim - kernel_size + 1  
#         #1, 105, 105
#         x = F.relu(self.bn1(self.conv1(x)))
#         # 64, 96, 96
#         x = F.max_pool2d(x, (2,2))
#         # 64, 48, 48
#         x = F.relu(self.bn2(self.conv2(x)))
#         # 128, 42, 42
#         x = F.max_pool2d(x, (2,2))
#         # 128, 21, 21
#         x = F.relu(self.bn3(self.conv3(x)))
#         # 128, 18, 18
#         x = F.max_pool2d(x, (2,2))
#         # 128, 9, 9
#         x = F.relu(self.bn4(self.conv4(x)))
#         # 256, 6, 6
#         return x

#     def forward(self, x1, x2):
#         x1 = self.convs(x1)

#         # Koch et al.
#         x1 = x1.view(-1, 256 * 6 * 6)
#         x1 = self.sigmoid(self.fc1(x1))
        
#         x2 = self.convs(x2)

#         # Koch et al.
#         x2 = x2.view(-1, 256 * 6 * 6)
#         x2 = self.sigmoid(self.fc1(x2))

#         x = torch.abs(x1 - x2)
#         x = self.fcOut(x)
#         return x


########### +++++++++++++++ ###########

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2