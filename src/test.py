from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from PIL import Image

import dataLoader as dl
import math

from misc import Utils

root_dir = '..\..\..\Dataset\MVTEC_AD'
model_path = 'models/best_contrastmodel_val_loss.pth'

def test(root_dir = root_dir):

    test_data = datasets.ImageFolder(root_dir + '/Test')
    # print(len(test_data))
    # print(test_data.imgs)
    # print(train_data.imgs) 
    # o/p will be tuple with each image path and the folder index
    # [('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\1.pgm', 0), ('..\\..\\..\\Dataset\\AT&T_Face\\Train\\s1\\10.pgm', 0)....]  

    siamese_test_dataset = dl.SNNTrain(imageFolderDataset = test_data,
                                            transform = transforms.Compose([transforms.Resize((128,128)),
                                                                            transforms.ToTensor()]),
                                                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                                                            should_invert = False)


    net = torch.load(model_path).cuda()

    net.eval()

    test_dataloader = DataLoader(siamese_test_dataset, num_workers=0, batch_size=1, shuffle=False)

    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    # data_len = len(siamese_test_dataset)

    # size_test = data_len - 1

    distances = []
    labels = []

    # Evaluation
    with torch.no_grad():
        for i in range(10):
            _, x1, label2 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)
            output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
            distance = F.pairwise_distance(output1, output2)
            Utils.imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(distance.item()))
            # # Uncomment to visualize testing data --> only for debugging
            # sample = iter(dataiter)
            # sampler = next(sample)
            # concatenated = torch.cat((sampler[0],sampler[1]), 0)
            # print(sampler[2].numpy()[0][0]) # 1 is for different class, 0 is for same class
            # Utils.imshow(torchvision.utils.make_grid(concatenated))

            # label = label.numpy()[0][0]

            # label = label.data.cpu().numpy()[0][0]
            # print(label)

            # else:
            #     output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
            #     distance = torch.sigmoid(F.pairwise_distance(output1, output2))
            #     if Config.bceLoss:
            #         distance = torch.sigmoid(distance)


            # labels.append(label)
            # distances.append(distance)

    # ##################
    # # data distibution analysis
    # sc_cont = 0
    # dc_cont = 0

    # sc_accum = 0
    # dc_accum = 0

    # sc_max = -1
    # sc_min = 99999

    # dc_max = -1
    # dc_min = 99999

    # for i in range(size_test):
    # # Count samples from same and different classes and finds the max / min distances
    #     if (labels[i] == 0):
    #         sc_cont += 1
    #         sc_accum += distances[i]

    #         if distances[i] < sc_min:
    #             sc_min = distances[i]
    #         if distances[i] > sc_max:
    #             sc_max = distances[i]
    #     else:
    #         dc_cont += 1
    #         dc_accum += distances[i]

    #         if distances[i] < dc_min:
    #             dc_min = distances[i]
    #         if distances[i] > dc_max:
    #             dc_max = distances[i]

    # sc_mean = sc_accum / sc_cont
    # dc_mean = dc_accum / dc_cont

    # # std dev
    # sc_std_accum = 0
    # dc_std_accum = 0
    # for i in range(size_test):
    #     if (labels[i] == 0):
    #         sc_std_accum += pow(distances[i] - sc_mean, 2)
    #     else:
    #         dc_std_accum += pow(distances[i] - dc_mean, 2)

    # sc_std = math.sqrt(sc_std_accum / sc_cont)
    # dc_std = math.sqrt(dc_std_accum / dc_cont)

    # print("Matching_samples: ", sc_cont)
    # print("Mismatching_samples: ", dc_cont)
    # print(" Mean std min max")
    # print("Match:", sc_mean.data.cpu().numpy()[0], sc_std, sc_min.data.cpu().numpy()[0],
    #         sc_max.data.cpu().numpy()[0])
    # print("Non-match:", dc_mean.data.cpu().numpy()[0], dc_std, dc_min.data.cpu().numpy()[0],
    #         dc_max.data.cpu().numpy()[0])

#     ##################
#     # ROC / metrics analysis
#     points = 20
#     thresholds = []

#     minn = min(sc_min, dc_min)
#     maxx = max(sc_max, dc_max)

#     print("min, max, step", minn.data.cpu().numpy()[0], ", ", maxx.data.cpu().numpy()[0], ", ",
#             ((maxx - minn) / points).data.cpu().numpy()[0], "\n")

#     for i in range(0, points + 1):
#         thresholds.append((minn) + (i * ((maxx - minn) / points)))

#     thresholds_str = "threshold: "
#     tp_str = "tp: "
#     tn_str = "tn: "
#     fp_str = "fp: "
#     fn_str = "fn: "
#     TPR_str = "TPR: "
#     FPR_str = "FPR: "

#     for thresh in thresholds:

#         tp = 0
#         tn = 0
#         fp = 0
#         fn = 0

#         for i in range(size_test):

#             if (labels[i] == 0 and distances[i] < thresh):
#                 tp += 1
#             elif (labels[i] == 0 and distances[i] > thresh):
#                 fn += 1
#             elif (labels[i] == 1 and distances[i] < thresh):
#                 fp += 1
#             elif (labels[i] == 1 and distances[i] > thresh):
#                 tn += 1

#         thresholds_str += str(thresh.data.cpu().numpy()[0]) + " "
#         tp_str += str(tp) + " "
#         tn_str += str(tn) + " "
#         fp_str += str(fp) + " "
#         fn_str += str(fn) + " "
#         TPR_str += str(tp / float(tp + fn)) + " "
#         FPR_str += str(fp / float(fp + tn)) + " "

#     print(thresholds_str)
#     print(tp_str)
#     print(tn_str)
#     print(fp_str)
#     print(fn_str)
#     print(TPR_str)
#     print(FPR_str)


# @staticmethod
def test_one():

    print("Testing one pair")

    query_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\good\000.png" # avoid special charcters
    target_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\contamination\000.png"

    query = Image.open(query_path)
    target = Image.open(target_path)

    transform = transforms.Compose([transforms.Resize((128,128)),
                                            transforms.ToTensor()])
                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                            # should_invert = False)
    # transform = transforms.Compose(
    #     [transforms.Resize((Config.im_w, Config.im_h)),  # transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor()])

    query = transform(query).unsqueeze(0)
    target = transform(target).unsqueeze(0)

    net = torch.load(model_path).cuda()

    net.eval()

    with torch.no_grad():

        output1, output2 = net(query.cuda(), target.cuda())
        concatenated = torch.cat((query, target), 0)
        distance = (F.pairwise_distance(output1, output2))
        Utils.imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(distance.item()))
        # distance = torch.sigmoid(distance)
        # print("S=", distance.item())

test_one()