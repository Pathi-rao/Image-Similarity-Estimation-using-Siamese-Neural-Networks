import shutil
import os
# select a random sample without replacement
from random import seed
from random import sample

# seed for random number generator
seed(13)

def split_dataset_into_3(path_to_dataset, train_ratio):
    """
    split the dataset in the given path into two subsets(validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :return:
    """
    # assert train_ratio
    valid_range = 0 < train_ratio <= 1
    assert valid_range, "train_ratio must be in range 0 to 1"

    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'new_train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'new_validation')
    # # dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        # dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir) # ..\..\Dataset\MVTEC_AD\Train\bottle
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))  # [209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        items = os.listdir(sub_dir) # a list of ALL image names

        # create a random sequence of ratio of train_ratio and use those as indexes to randomly select images
        trainratio = round((sub_dir_item_cnt[i] + 1)* train_ratio)
        # prepare a sequence
        sequence = [i for i in range(sub_dir_item_cnt[i])]
        # select a subset without replacement
        train_subset = sample(sequence, trainratio) # percentage of random numbers of size trainratio
        
        for i in range(len(sequence)):
            # if the number exists in our randomly generated train_subset then copy it to train folder
            if i in train_subset:
                if not os.path.exists(dir_train_dst):
                    os.makedirs(dir_train_dst)
                source_file = os.path.join(sub_dir, items[i])
                dst_file = os.path.join(dir_train_dst, items[i])
                shutil.copyfile(source_file, dst_file)

            # if it's not, then copy it to validation folder
            else:
                if not os.path.exists(dir_valid_dst):
                    os.makedirs(dir_valid_dst)
                source_file = os.path.join(sub_dir, items[i])
                dst_file = os.path.join(dir_valid_dst, items[i])
                shutil.copyfile(source_file, dst_file)

    return

# split_dataset_into_3('..\..\Dataset\MVTEC_AD\Train' , 0)