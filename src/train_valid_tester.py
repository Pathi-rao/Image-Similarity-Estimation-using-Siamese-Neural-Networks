import torch
import torchvision

import datahandler as dh
from misc import Utils

"""

The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image. 
1 indiciates dissimilar, and 0 indicates similar.
"""

root_dir = '../../Dataset/MVTEC_AD'
trainbatchsize = 4
validbatchsize = 4
testbatchsize = 1


def display_samples (show_train = True, show_valid = True):

    train_loader, valid_loader, test_loader = dh.pre_processor(root_dir=root_dir, trainbatchsize=trainbatchsize, 
                                                validbatchsize=validbatchsize, testbatchsize=testbatchsize)

    if show_valid:
        valid_dataiter = iter(valid_loader)
        # print(dataiter)
        example_batch = next(valid_dataiter)
        # print(example_batch)
        concatenated = torch.cat((example_batch[0],example_batch[1]),0)
        print(example_batch[2].numpy()) # 1 is for different class, 0 is for same class
        Utils.imshow(torchvision.utils.make_grid(concatenated, nrow=4))

    if show_train:
        train_dataiter = iter(train_loader)
        # print(dataiter)
        example_batch = next(train_dataiter)
        # print(example_batch)
        concatenated = torch.cat((example_batch[0],example_batch[1]),0)
        print(example_batch[2].numpy()) # 1 is for different class, 0 is for same class
        Utils.imshow(torchvision.utils.make_grid(concatenated, nrow=4))

    pass

# display_samples (show_train = True, show_valid = True)

if __name__ == '__main__': # to avoid multi-processing error

    display_samples(show_train = True, show_valid = True)

