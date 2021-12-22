import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt
import argparse

from loss import ContrastiveLoss
import datahandler as dl
from misc import Utils
from model import SiameseNetwork

# parser = argparse.ArgumentParser(description="Siamese CNN")
# parser.add_argument('--epochs', metavar = 'e', type = int, required = True)
# parser.add_argument('--lr', metavar = 'l', type = float, required = True)
# parser.add_argument('--dropout', metavar = 'd', type = float, required = True)
# parser.add_argument('--saved_data', metavar = 's', type = int, default= 1, required = False)
# args = vars(parser.parse_args())

root_dir = '..\..\Dataset\MVTEC_AD'
epochs = 50
lear_rate = 0.0005
trainbatchsize = 4
validbatchsize = 4

train_loader, valid_loader = dl.pre_processor(root_dir=root_dir, trainbatchsize=trainbatchsize, 
                                                validbatchsize=validbatchsize)

net = SiameseNetwork().cuda()
# net.train()
criterion = ContrastiveLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr = lear_rate )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

train_losses = []
val_losses = []

best_val = 0.1

start_time = time.time()

for epoch in range( epochs):

    net.train()
    running_loss = 0

    for i, data in enumerate(train_loader, 0): # giving the argument 0 starts the counter from 0
        img0, img1 , label = data

        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        # reset the gradients
        optimizer.zero_grad()

        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)

        loss_contrastive.backward()                             # backward pass
        optimizer.step()                                        # update weights

        running_loss += loss_contrastive.item()

    # print('My loss... ', running_loss /img0.shape[0] )
    # print('cjv loss....', running_loss/i)
    train_losses.append(running_loss /len(train_loader))

    if epoch % 2 == 0:
        val_loss = 0
        net.eval()
        with torch.no_grad():
            for i , data in enumerate(valid_loader, 0):
                img_0, img_1, label_ = data
                img_0, img_1 , label_ = img_0.cuda(), img_1.cuda() , label_.cuda()
                output1, output2 = net(img_0, img_1)
                # distance = torch.sigmoid(F.pairwise_distance(output1, output2))
                loss_val = criterion(output1, output2, label_)
                val_loss += loss_val.item() 

        # print('validation image shape is:   ', img_0.shape[0])
        val_losses.append(val_loss/len(valid_loader))

        print('Epoch : ',epoch, "\t Train loss: {:.2f}".format(running_loss/len(train_loader)),
            "\t Validation loss: {:.2f}".format(val_loss/len(valid_loader)))

        if val_loss/len(valid_loader) < best_val:
            best_val = val_loss/len(valid_loader)
            PATH = 'models/best_model_val_loss.pth'
            torch.save(net, PATH)
            print()
            print("Saved best model at epoch:  ", epoch)
            print()

    scheduler.step()

print('It took {} seconds to train the model.. '.format(time.time() - start_time))



# plot and save the losses
fig = plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label = "train")
plt.plot(val_losses, label = "val")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
fig.savefig('new_Train_&_Val_loss.png')


