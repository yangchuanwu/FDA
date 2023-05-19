from __future__ import absolute_import
import torch
import time
import numpy as np
from util.utils import AverageMeter, Logger
from util.metric import compute_acc


def train(args, epoch, model, criterion, optimizer, train_loader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, label, _) in enumerate(train_loader):
        if use_gpu:
            imgs, label = imgs.cuda(), label.cuda()

        data_time.update(time.time() - end)
        optimizer.zero_grad()

        _, outputs = model(imgs)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), label.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses))


def validate(args, model, val_loader, use_gpu):
    batch_time = AverageMeter()
    model.eval()
    accs = []

    with torch.no_grad():
        for batch_idx, (imgs, label, _) in enumerate(val_loader):
            if use_gpu:
                imgs = imgs.cuda()
                label = label.cuda()

            end = time.time()
            _, outputs = model(imgs)
            batch_time.update(time.time() - end)
            pred = outputs > 0.5

            acc = compute_acc(label, pred)
            accs.append(acc)

    return np.mean(accs)



