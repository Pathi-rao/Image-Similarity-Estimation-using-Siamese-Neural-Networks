from random import triangular
import torch
from torch import optim
from torch.utils import tensorboard
import torchvision
import os
import time
import matplotlib.pyplot as plt
from misc import Utils

from loss import ContrastiveLoss
import datahandler as dl
from model import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = '..\..\Dataset\MVTEC_AD'
epochs = 100
lear_rate = 0.0005
trainbatchsize = 4
validbatchsize = 4
testbatchsize = 1
log_dir='test_logs'

train_loader, valid_loader, test_loader = dl.pre_processor(root_dir=root_dir, trainbatchsize=trainbatchsize, 
                                                validbatchsize=validbatchsize,
                                                testbatchsize=testbatchsize)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = tensorboard.SummaryWriter(log_dir)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr = lear_rate )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


train_loss_history = [] 
val_loss_history = [] 

iteration_number = 0
val_iter_number = 0

best_val = 0.1

start_time = time.time()


for epoch in range(epochs):
    # print(epoch)
    net.train()
    running_loss = 0

    for i, data in enumerate(train_loader, 0): # giving the argument 0 starts the counter from 0
        img0, img1 , label = data

        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()

        optimizer.zero_grad()               # reset the gradients

        # output1, output2 = net(img0, img1)
        scores =  net(img0, img1)
        loss_contrastive = criterion(scores, label) # find the loss

        loss_contrastive.backward()         # backward pass
        optimizer.step()                    # update weights

        running_loss += loss_contrastive.item()

    writer.add_scalar('train_loss/epoch', running_loss/len(train_loader), epoch)
    print(f"Epoch number:  {epoch}\t loss: {running_loss/len(train_loader)}")

    if epoch % 2 == 0: # validate for every 2 epochs
        train_loss_history.append(running_loss/len(train_loader))
        net.eval() #set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for i , data in enumerate(valid_loader, 0):
                img_0, img_1, label_ = data
                img_0, img_1 , label_ = img_0.cuda(), img_1.cuda() , label_.cuda()
                scores = net(img_0, img_1)
                loss_val = criterion(scores, label_)
                val_loss += loss_val.item()
                val_loss_history.append(val_loss/len(valid_loader))

            writer.add_scalar('valid_loss/epoch', val_loss, epoch)

        if loss_val.item() < best_val and best_val != 0.000:
            # print("loss is ... ", loss_val.item())
            best_val = loss_val.item()
            print("best val loss is.. {}\t and saved at epoch {}\t ".format(best_val, epoch))
            PATH = '../models/best_model_val_loss.pth'
            torch.save(net.state_dict(), PATH)

    scheduler.step()

print('It took {} seconds to train the model.. '.format(time.time() - start_time))
#It took 12678.214158058167 seconds to train the model..



