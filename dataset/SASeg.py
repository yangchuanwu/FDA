from __future__ import print_function, absolute_import
import os
import torch.utils.data as data
import numpy as np

from skimage import io


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = io.imread(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def img_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    img -= mean
    img /= std
    return img


class SASeg(data.Dataset):
    """

    """

    def __init__(self, data_root, transform=None, max_iters=None, split='train'):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.index = self.load_index()
        if not max_iters==None:
            self.index = self.index * int(np.ceil(float(max_iters) / len(self.index)))

    def load_index(self):
        index_path = os.path.join(self.data_root, self.split + '.txt')
        f = open(index_path, 'r')
        index = f.read().splitlines()
        f.close()

        return index

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        idx = self.index[item]
        se_path = os.path.join(self.data_root, 'SE/', idx)
        hd_path = os.path.join(self.data_root, 'HD/', idx)
        label_path = os.path.join(self.data_root, 'label/', idx)
        img_se = read_image(se_path).astype('float32')
        img_hd = read_image(hd_path).astype('float32')
        label = read_image(label_path)
        label = (label > 0).astype('float32')

        img_se = np.expand_dims(img_norm(img_se), axis=0)
        img_hd = np.expand_dims(img_norm(img_hd), axis=0)
        img = np.concatenate((img_se, img_hd), axis=0)
        label = np.expand_dims(label, axis=0)
        return img, label, idx

    def __len__(self):
        return len(self.index)


class SASeg_Feat(data.Dataset):
    def __init__(self, data_root, transform=None, max_iters=None, split='train'):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.index = self.load_index()
        if not max_iters==None:
            self.index = self.index * int(np.ceil(float(max_iters) / len(self.index)))

    def load_index(self):
        index_path = os.path.join(self.data_root, self.split + '.txt')
        f = open(index_path, 'r')
        index = f.read().splitlines()
        f.close()

        return index

    def __getitem__(self, item):
        idx = self.index[item]
        feat_path = os.path.join(self.data_root, 'feat/', idx[:4] + '.npz')
        read_data = np.load(feat_path)
        feat = read_data['arr_0']
        output = read_data['arr_1']
        label = read_data['arr_2']
        return feat, output, label

    def __len__(self):
        return len(self.index)