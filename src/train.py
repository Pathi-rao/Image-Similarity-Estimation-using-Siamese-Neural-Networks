import torch
from torch import optim
import time
import matplotlib.pyplot as plt

from loss import ContrastiveLoss
import datahandler as dl
from model import SiameseNetwork

root_dir = '..\..\Dataset\MVTEC_AD'
epochs = 50
lear_rate = 0.0005
trainbatchsize = 4
validbatchsize = 4
testbatchsize = 1

train_loader, valid_loader, test_loader = dl.pre_processor(root_dir=root_dir, trainbatchsize=trainbatchsize, 
                                                validbatchsize=validbatchsize,
                                                testbatchsize=testbatchsize)

net = SiameseNetwork().cuda()
# net.train()
criterion = ContrastiveLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr = lear_rate )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

train_loss_history = [] 
val_loss_history = [] 

best_val = 0.1

start_time = time.time()

for epoch in range(epochs):
    # print(epoch)
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

    if epoch % 2 == 0: # validate for every 2 epochs
        train_loss_history.append(running_loss/len(train_loader))
        # print(epoch)
        print("Epoch number {}\t Train loss {}\t".format(epoch, running_loss/len(train_loader)))
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i , data in enumerate(valid_loader, 0):
                img_0, img_1, label_ = data
                img_0, img_1 , label_ = img_0.cuda(), img_1.cuda() , label_.cuda()
                output1, output2 = net(img_0, img_1)
                loss_val = criterion(output1, output2, label_)
                val_loss += loss_val.item()
                val_loss_history.append(val_loss/len(valid_loader))

        if loss_val.item() < best_val:
            # print("loss is ... ", loss_val.item())
            best_val = loss_val.item()
            print("best val loss is.. {}\t and saved at epoch {}\t ".format(best_val, epoch))
            PATH = '../models/best_model_val_loss.pth'
            torch.save(net.state_dict(), PATH)

    scheduler.step()

print('It took {} seconds to train the model.. '.format(time.time() - start_time))



# plot and save the losses
fig = plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_loss_history, label = "train")
plt.plot(val_loss_history, label = "val")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
fig.savefig('Train_Val_loss.png')


