import os
import argparse
import time
import datetime
import torch
import xlwt

from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.SASeg import SASeg
from torchvision import transforms as T
from model.unet_plus_plus import NestNet
import numpy as np
from util.metric import compute_acc, mean_iou


def main(args):
    use_gpu = torch.cuda.is_available()

    print("Initializing dataset {}".format(args.dataset))

    transform_test = T.Compose([
        T.ToTensor(),
    ])
    test_loader = DataLoader(
        SASeg(args.data_root, transform=transform_test, split='test'),
        batch_size=args.test_batch, num_workers=args.workers, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = NestNet(in_channels=2, n_classes=1)

    # for dir in ['6', '5', '4', '3', '2']:
    #     for name in ['50', '45', '40', '35', '30', '25', '20', '15', '10', '5']:
    #         model_path = os.path.join('./log_900_1000/snapshots/', dir + '/GTA5_' + name + '000.pth')
    # model_path = os.path.join('./log_1000_900/snapshots/6/GTA5_30000.pth')
    # print('Loading model from {}'.format(model_path))
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)

    model_path = os.path.join(args.save_dir, 'best_model.pth.tar')
    print('Loading model from {}'.format(model_path))
    checkpoint = torch.load(model_path)
    print('acc:{} at epoch {}'.format(checkpoint['acc'], checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model = model.cuda()

    print("==> Start testing")

    start_time = time.time()
    model.eval()
    accs = []
    mIoUs = []
    #
    wb = xlwt.Workbook()
    sheet = wb.add_sheet('sheet2')
    # save_path = os.path.join(args.data_root, 'feat')
    with torch.no_grad():
        for batch_idx, (img, label, idx) in enumerate(test_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            feat, output = model(img)
            # for i in range(len(idx)):
            #     np.savez(os.path.join(save_path, idx[i][:4]),
            #              feat[i].cpu().numpy(), output[i].cpu().numpy(), label[i].cpu().numpy())

            pred = output > 0.5
            acc = compute_acc(label, pred)
            mIoU = mean_iou(pred, label)
            sheet.write(batch_idx, 0, str(round(acc, 3)))
            sheet.write(batch_idx, 1, str(round(mIoU, 3)))
            sheet.write(batch_idx, 2, idx)
            accs.append(acc)
            mIoUs.append(mIoU)
    wb.save('result/test.xls')
    total_acc = np.mean(accs)
    total_mIoU = np.mean(mIoUs)
    test_time = round(time.time() - start_time)
    test_time = str(datetime.timedelta(seconds=test_time))
    print("==> Acc: {:.1%}, mIoU: {:.1%}, test time (h:m:s): {}.".format(total_acc, total_mIoU, test_time))


def parse_args():
    parser = argparse.ArgumentParser(description='test for segmentation')
    # data
    parser.add_argument('--dataset', default='super-alloy W-900')
    parser.add_argument('--data_root', type=str, default='../Data/',
                        help="root path to data directory")
    # testing
    parser.add_argument('--test-batch', default=2, type=int,
                        help="test batch size")
    parser.add_argument('--workers', default=0, type=int)
    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='Unet++',
                        help="model")
    parser.add_argument('--save-dir', type=str, default='./log_1000_900')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
