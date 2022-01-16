import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataLoader as dl

"""

A function which creates train and valid dataloaders that can be iterated over.
To do so, you need to structure your data as follows:
root_dir
    |_train
        |_class_1
            |_xxx.png
        .....
        .....    
        |_class_n
            |_xxx.png
    |_validation
        |_class_1
            |_xxx.png
        .....
        .....
        |_class_n
            |_xxx.png
that means that each class has its own directory.
By giving this structure, the name of the class will be taken by the name of the folder!

Parameters
----------
root_dir: (str) "Path to where the data is"
trainbatchsize: (int) batch size for training
validbatchsize: (int) batch size for validation
testbatchsize: (int) batch size for testing the model
"""

def pre_processor(root_dir, trainbatchsize, validbatchsize, testbatchsize):

    train_data = datasets.ImageFolder(root_dir + '/New_train')
    test_data = datasets.ImageFolder(root_dir + '/New_test')

    siamese_train_dataset = dl.SNNTrain(imageFolderDataset = train_data,
                                            transform = transforms.Compose([transforms.Resize((105,105)),
                                                                            transforms.ToTensor()]),
                                                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                                                            should_invert = False)
    siamese_test_dataset = dl.SNNTest(imageFolderDataset = test_data,
                                            transform = transforms.Compose([transforms.Resize((105,105)),
                                                                            transforms.ToTensor()]),
                                                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                                                            should_invert = False)
    # Train_valid split                                                                        
    train_len = int(0.8*len(siamese_train_dataset)) # 80:20 split
    valid_len = len(siamese_train_dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(siamese_train_dataset, [train_len, valid_len])

    # create the dataloaders
    train_loader = DataLoader(train_set, batch_size = trainbatchsize,
                                            shuffle = True
                                            )
    valid_loader =  DataLoader(val_set, batch_size = validbatchsize,
                                            shuffle = False) # shuffle doesn't matter during validation and testing 
    test_loader = DataLoader(siamese_test_dataset, batch_size = testbatchsize,
                                            shuffle = False)

    return train_loader , valid_loader, test_loader
