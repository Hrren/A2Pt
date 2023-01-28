"""
import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from data.imagenet_coop import ImageNet


#@DATASET_REGISTRY.register()
class ImageNetV2(DatasetBase):
    mageNetV2.

    This dataset is used for testing only.
    

    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)
        self.classnames = classnames
        data = self.read_data(classnames)

        super().__init__(train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
"""

import torchvision
import numpy as np

import os

from copy import deepcopy
from config import imagenet_v2_val

def create_val_img_folder(root):
    '''
    This method is responsible for separating validation images into separate sub folders
    Run this before running TinyImageNet experiments

    :param root: Root dir for TinyImageNet, e.g /work/sagar/datasets/tinyimagenet/tiny-imagenet-200/
    '''
    dataset_dir = os.path.join(root)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


class ImageNetV2(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetV2, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label, index, path = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(20)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_imagenet_v2_datasets(train_transform, test_transform, train_classes=range(20),
                       open_set_classes=range(20, 200), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    # Init train dataset and subsample training classes
    #train_dataset_whole = ImageNetV2(root=tin_train_root_dir, transform=train_transform)
    #train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    #train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    #val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = ImageNetV2(root=imagenet_v2_val, transform=test_transform)
    #test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    #test_dataset_unknown = ImageNetV2(root=tin_val_root_dir, transform=test_transform)
    #test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    #if balance_open_set_eval:
    #    test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    #train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    #val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': test_dataset_known,
        'val': test_dataset_known,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_known,
    }

    return all_datasets

if __name__ == '__main__':

    x = get_tiny_image_net_datasets(None, None, balance_open_set_eval=False, split_train_val=False)
    print([len(v) for k, v in x.items()])
    debug = 0

