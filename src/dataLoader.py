import numpy as np
from PIL import Image
import PIL.ImageOps
import random

import torch
from torch.utils.data import Dataset


"""
Class which creates image pairs. 
"""

class SNNTrain(Dataset):
    
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self, index):

        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 

        if should_get_same_class: # Enter the loop if value is 1
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        # convert images to RGB format
        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple[0]).convert('RGB')

        # convert images to grayscale if passed "L". 
        # img0 = Image.open(img0_tuple[0]).convert("L")
        # img1 = Image.open(img1_tuple[0]).convert("L")
        
        if self.should_invert:

            """ 
            The ImageOps module contains a number of 'ready-made' image processing operations
            convert series of images drawn as white on black background images to 
            images where white and black are inverted (as negative)
            """
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0) # apply the given tranformation(when instantiating the class).
            img1 = self.transform(img1)
        # return images and label
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32)) 
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SNNTest(Dataset):
    
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self, index):

        # Pick the same image twice
        img0_tuple = self.imageFolderDataset.imgs[index // 2]

        should_get_same_class = index % 2

        if should_get_same_class: # Enter the loop if value is 1
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        # convert images to RGB format
        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple[0]).convert('RGB')

        # convert images to grayscale if passed "L".
        # img0 = Image.open(img0_tuple[0]).convert("L")
        # img1 = Image.open(img1_tuple[0]).convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0) # apply the given tranformation(when instantiating the class).
            img1 = self.transform(img1)

        # return images and label
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32)), img0_tuple[1] 
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs) * 2

# # creates n-way one shot learning evaluation
# class NWayOneShotEvalSet(Dataset):
#     def __init__(self, categories, root_dir, setSize, numWay, transform=None):
#         self.categories = categories
#         self.root_dir = root_dir
#         self.setSize = setSize
#         self.numWay = numWay
#         self.transform = transform
#     def __len__(self):
#         return self.setSize
#     def __getitem__(self, idx):
#         # find one main image
#         category = random.choice(categories)
#         character = random.choice(category[1])
#         imgDir = root_dir + category[0] + '/' + character
#         imgName = random.choice(os.listdir(imgDir))
#         mainImg = Image.open(imgDir + '/' + imgName)
#         # print(imgDir + '/' + imgName)
#         if self.transform:
#             mainImg = self.transform(mainImg)
        
#         # find n numbers of distinct images, 1 in the same set as the main
#         testSet = []
#         label = np.random.randint(self.numWay)
#         for i in range(self.numWay):
#             testImgDir = imgDir
#             testImgName = ''
#             if i == label:
#                 testImgName = random.choice(os.listdir(imgDir))
#             else:
#                 testCategory = random.choice(categories)
#                 testCharacter = random.choice(testCategory[1])
#                 testImgDir = root_dir + testCategory[0] + '/' + testCharacter
#                 while testImgDir == imgDir:
#                     testImgDir = root_dir + testCategory[0] + '/' + testCharacter
#                 testImgName = random.choice(os.listdir(testImgDir))
#             testImg = Image.open(testImgDir + '/' + testImgName)
#             if self.transform:
#                 testImg = self.transform(testImg)
#             testSet.append(testImg)
#         # plt.imshow()
#         return mainImg, testSet, torch.from_numpy(np.array([label], dtype = int))