import os
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.optim as optim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.SASeg import SASeg, SASeg_Feat
from torchvision import transforms as T
from model.unet_plus_plus import NestNet
from util.utils import save_checkpoint
from util.optimizers import init_optim
from util.function import train, validate
from model.loss import FCD_loss
from model.discriminator import FCDiscriminator


SOURCE_DATA_ROOT = '../Data1/W-900/'
TARGET_DATA_ROOT = '../Data/'
NUM_TARGET = '5'
RESTORE_FROM = ''#'./log_900_1000/best_model.pth.tar'
SAVE_DIR = 'nlog_900_1000/log_pf3_/'

TRAIN_BATCH = 2
TEST_BATCH = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
LEARNING_RATE_D1 = 1e-4
LEARNING_RATE_D2 = 1e-5
LAMBDA_D2 = 0.01
NUM_ITERS = 10000
POWER = 0.9
EVAL_STEP = 200
USE_PROB = True
USE_FEAT = True
USE_LEVEL = 3


def train_source(args):
    use_gpu = torch.cuda.is_available()

    print("Initializing dataset {}".format(args.dataset))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # data augmentation
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_val = T.Compose([T.ToTensor()])

    train_loader = DataLoader(
        SASeg(args.source_data_root, transform=transform_train, split='test'),
        batch_size=args.train_batch, num_workers=args.workers, drop_last=False,
    )

    val_loader = DataLoader(
        SASeg(args.source_data_root, transform=transform_val, split='val'),
        batch_size=args.test_batch, num_workers=args.workers, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = NestNet(in_channels=2, n_classes=1)

    # criterion
    criterion = nn.BCELoss()

    optimizer = init_optim(args.optim, model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if args.pre_trained:
        print("Loading checkpoint from '{}'".format(args.pre_trained))
        checkpoint = torch.load(args.pre_trained)
        model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0

    print("==> Start training")
    model.train()
    for epoch in range(200):
        start_train_time = time.time()

        train(args, epoch, model, criterion, optimizer, train_loader, use_gpu)

        train_time += round(time.time() - start_train_time)
        scheduler.step()

        print("==> Validate")
        acc = validate(args, model, val_loader, use_gpu)

        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'acc': acc,
                'epoch': epoch,
            }, False, os.path.join(args.save_dir, 'best_model_.pth.tar'))

        print("==> Acc {:.1%} at epoch {}, with lr {}".format(acc, epoch+1, optimizer.param_groups[0]['lr']))
        print("==> Best Acc {:.1%}, achieved at epoch {}".format(best_acc, best_epoch))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        _train_time = str(datetime.timedelta(seconds=train_time))
        print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, _train_time))


def parse_args():
    parser = argparse.ArgumentParser(description='train for segmentation')
    # data
    parser.add_argument('--dataset', default='superalloy')
    parser.add_argument('--source_data_root', type=str, default=SOURCE_DATA_ROOT,
                        help="root path to target data directory")
    parser.add_argument('--target_data_root', type=str, default=TARGET_DATA_ROOT,
                        help="root path to target data directory")
    parser.add_argument('--num_target', default=NUM_TARGET, type=str,
                        help="number of labeled sample in target dataset")
    # source training
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument('--gamma', default=0.5, type=float,
                        help="learning rate decay")
    parser.add_argument('--stepsize', default=50, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    # training
    parser.add_argument('--num-iters', default=NUM_ITERS, type=int,
                        help="maximum iters to run")
    parser.add_argument('--train-batch', default=TRAIN_BATCH, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=TEST_BATCH, type=int,
                        help="test batch size")
    parser.add_argument('--optim', default='adam', type=str,
                        help="optimizer type")
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument('--stepsizet', default=500, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--learning-rate-D1', type=float, default=LEARNING_RATE_D1,
                        help="Base learning rate for discriminator 1.")
    parser.add_argument('--learning-rate-D2', type=float, default=LEARNING_RATE_D2,
                        help="Base learning rate for discriminator 2.")
    parser.add_argument('--lambda-D2', type=float, default=LAMBDA_D2,
                        help="Lambda for discriminator 2.")
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument('--weight-decay', default=5e-05, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--power', type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    # Parameter
    parser.add_argument('--use-feat', default=USE_FEAT, type=bool,
                        help="Whether feature alignment is used")
    parser.add_argument('--use-prob', default=USE_PROB, type=bool,
                        help="Whether probability alignment is used")
    parser.add_argument('--levels', default=USE_LEVEL, type=int,
                        help="Which layers of features are aligned")
    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='Unet++',
                        help="model")
    parser.add_argument('--resume', type=str, default='',
                        metavar='PATH')
    parser.add_argument('--pre-trained', type=str, default=RESTORE_FROM,
                        metavar='PATH')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--eval-step', type=int, default=EVAL_STEP,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    train_source(args)