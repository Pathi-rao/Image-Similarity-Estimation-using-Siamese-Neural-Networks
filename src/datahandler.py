import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataLoader as dl

def pre_processor(root_dir, trainbatchsize, validbatchsize):

    train_data = datasets.ImageFolder(root_dir + '/Train')
    # print(train_data.imgs) 
    # o/p will be tuple with each image path and the folder index
    # [('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\1.pgm', 0), ('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\10.pgm', 0)....]  

    siamese_train_dataset = dl.SNNTrain(imageFolderDataset = train_data,
                                            transform = transforms.Compose([transforms.Resize((128,128)),
                                                                            transforms.ToTensor()]),
                                                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                                                            should_invert = False)

    # print(len(siamese_train_dataset))
    # Train_valid split                                                                        
    train_len = int(0.8*len(siamese_train_dataset)) # 80:20 split
    valid_len = len(siamese_train_dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(siamese_train_dataset, [train_len, valid_len])

    # create the dataloaders
    train_loader = DataLoader(train_set, batch_size = trainbatchsize,
                                            shuffle = True)
    valid_loader =  DataLoader(val_set, batch_size = validbatchsize,
                                            shuffle = False) # shuffle doesn't matter during validation and testing

    # print('Data is splitted into {} training_samples and {} validation samples'.format(len(train_loader), len(valid_loader)))
    
    return train_loader , valid_loader


# debug
train_loader, valid_loader = pre_processor(root_dir='..\..\..\Dataset\MVTEC_AD', trainbatchsize=4, 
                                                validbatchsize=4)