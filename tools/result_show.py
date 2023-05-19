import os

import cv2
import numpy as np
import torch
import sys
from skimage import io

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.unet_plus_plus import NestNet


def img_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    img -= mean
    img /= std
    return img


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


def load_data(data_root, idx):
    se_path = os.path.join(data_root, 'SE/', idx)
    hd_path = os.path.join(data_root, 'HD/', idx)
    label_path = os.path.join(data_root, 'label/', idx)
    img_se = read_image(se_path).astype('float32')
    img_hd = read_image(hd_path).astype('float32')
    label = read_image(label_path)
    label = (label > 0).astype('float32')

    img_se = np.expand_dims(img_norm(img_se), axis=0)
    img_hd = np.expand_dims(img_norm(img_hd), axis=0)
    img = np.concatenate((img_se, img_hd), axis=0)
    label = np.expand_dims(label, axis=0)

    img = torch.from_numpy(img).unsqueeze(0)
    label = torch.from_numpy(label)

    return img, label


data_root = '../Data1/W-900/'
data_name = '0011.tif'
save_dir = './log_1000_900/log_pf123'

print("Initializing model: test for segmentation")
model = NestNet(in_channels=2, n_classes=1)

print('load data {}'.format(data_name))
img, label = load_data(data_root, data_name)

model_path = os.path.join('./log_1000_900/snapshots/3/GTA5_50000.pth')
print('Loading model from {}'.format(model_path))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

# model_path = os.path.join(save_dir, 'best_model_6.pth.tar')
# print('Loading model from {}'.format(model_path))
# checkpoint = torch.load(model_path)
# print('acc:{} at epoch {}'.format(checkpoint['acc'], checkpoint['epoch']))
# model.load_state_dict(checkpoint['state_dict'])

model = model.cuda()
model.eval()

img = img.cuda()
label = label.cuda()

_, output = model(img)
output = output.squeeze()

output = output.detach().cpu().numpy()

print('save result')
im_grey = np.where(output < 0.5, 0, 255)
data = np.array(im_grey, dtype='uint8')
io.imsave('./result/gan2.jpg', data)

im_grey2 = (output - output.min()) / (output.max() - output.min()) * 255
data2 = np.array(im_grey2, dtype='uint8')
io.imsave('./result/gan2_.jpg', data2)
print('successful')



