import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torchvision

from PIL import Image

from misc import Utils

root_dir = '..\..\..\Dataset\MVTEC_AD'
# model_path = r"..\models\model_at_epoch_10.pth" # old model

model_path = r"..\models\best_model_val_loss.pth" # updated model

query_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\good\000.png" # avoid special charcters
# target_path = r"D:\Github\Dataset\MVTEC_AD\Test\cable\bent_wire\000.png"
# target_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\contamination\000.png"
# target_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\good\001.png"
# target_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\good\000.png"
target_path = r"D:\Github\Dataset\MVTEC_AD\Test\bottle\broken_large\000.png"
# target_path = r"D:\Github\Dataset\MVTEC_AD\Test\hazelnut\crack\000.png"


def test_one(query_path, target_path):

    print("Testing one pair")

    query = Image.open(query_path)
    target = Image.open(target_path)

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((105,105)),
                                    transforms.ToTensor()])
                                            # transforms.Normalize([0.4318, 0.4012, 0.3913], [0.2597, 0.2561, 0.2525])]),
                                            # should_invert = False)
    # transform = transforms.Compose(
    #     [transforms.Resize((Config.im_w, Config.im_h)),  # transforms.Grayscale(num_output_channels=3),
    #     transforms.ToTensor()])

    query = transform(query).unsqueeze(0)
    target = transform(target).unsqueeze(0)

    net = torch.load(model_path).cuda()
    # print(net)
    

    with torch.no_grad():
        net.eval()
        output1, output2 = net(query.cuda(), target.cuda())
        concatenated = torch.cat((query, target), 0)
        distance = (F.pairwise_distance(output1, output2))
        Utils.imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(distance.item()))
        # sigmoid_distance = torch.sigmoid(distance)
        # print("S=  {:.2f}".format(sigmoid_distance.item()))

test_one(query_path, target_path)

