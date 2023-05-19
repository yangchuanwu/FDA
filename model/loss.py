# -*- coding:utf-8 -*-
import random

import torch
import torch.nn as nn
# import numpy as np
#
# from lap import lapjv
from kmeans_pytorch import kmeans
# from torch.autograd import Variable
# from .layers import SinkhornDistance


def mat_dist(mat1, mat2):
    sq1 = mat1**2
    sum_sq1 = torch.sum(sq1, dim=1).unsqueeze(1)  # m->[m, 1]
    sq2 = mat2**2
    sum_sq2 = torch.sum(sq2, dim=1).unsqueeze(0)  # n->[1, n]
    return torch.sqrt(sum_sq1 + sum_sq2 - 2*mat1.mm(mat2.t()))


class FCD_loss(nn.Module):
    def __init__(self, feat_dim=32, ndf=32):
        super(FCD_loss, self).__init__()
        self.conv1 = nn.Conv2d(feat_dim, ndf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if len(x.size()) > 4:
            x = x.sum(dim=1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class MMD_loss(nn.Module):
    def __init__(self, start_layer=0, sample_num=100):
        super(MMD_loss, self).__init__()
        self.start_layer = start_layer
        self.sample_num = sample_num

    def forward(self, source_feat, target_feat, *labels):
        losses = 0
        # sinkhorn = SinkhornDistance(eps=1, max_iter=100, reduction=None)
        _, num_layer, dim, _, _ = source_feat.size()
        s_feats = source_feat.permute(0, 3, 4, 1, 2)
        t_feats = target_feat.permute(0, 3, 4, 1, 2)

        if labels:
            source_out, source_label, target_label = labels
            s0 = s_feats[(source_label.squeeze(1) == 0) & (source_out.squeeze(1) <= 0.5)]
            s1 = s_feats[(source_label.squeeze(1) == 1) & (source_out.squeeze(1) > 0.5)]

            t0 = t_feats[target_label.squeeze(1) == 0]
            t1 = t_feats[target_label.squeeze(1) == 1]
            # sample the data
            s0 = s0[:self.sample_num * (len(s0)//self.sample_num)]
            s0 = s0.view(-1, self.sample_num, num_layer, dim)
            s0 = torch.mean(s0, dim=0)

            s1 = s1[:self.sample_num * (len(s1)//self.sample_num)]
            s1 = s1.view(-1, self.sample_num, num_layer, dim)
            s1 = torch.mean(s1, dim=0)

            t0 = t0[:self.sample_num * (len(t0)//self.sample_num)]
            t0 = t0.view(-1, self.sample_num, num_layer, dim)
            t0 = torch.mean(t0, dim=0)

            t1 = t1[:self.sample_num * (len(t1)//self.sample_num)]
            t1 = t1.view(-1, self.sample_num, num_layer, dim)
            t1 = torch.mean(t1, dim=0)

            for i in range(self.start_layer, num_layer):
                # l0, _, _ = sinkhorn(s0[sample_list, i], t0[sample_list, i])
                # l1, _, _ = sinkhorn(s1[sample_list, i], t1[sample_list, i])
                l00 = self.mmd(s0[:, i], t0[:, i])
                l11 = self.mmd(s1[:, i], t1[:, i])
                # l0 = self.hausdorff_distance(s0[sample_list, i], t0[sample_list, i])
                # l1 = self.hausdorff_distance(s1[sample_list, i], t1[sample_list, i])
                loss = l00 + l11 #- l01 - l10 + 16
                losses += loss / 2
        else:
            s_feats = s_feats.view(-1, num_layer, dim)
            idx = torch.randperm(len(s_feats))
            s = s_feats[idx[:self.sample_num]]
            t_feats = t_feats.view(-1, num_layer, dim)
            idx = torch.randperm(len(t_feats))
            t = t_feats[idx[:self.sample_num], :]

            for i in range(self.start_layer, num_layer):
                losses += self.mmd(s[:, i], t[:, i])

        return losses / (num_layer - self.start_layer)

    def hausdorff_distance(self, source, target):
        dist = mat_dist(source, target)
        h1 = torch.max(torch.min(dist, dim=0)[0])
        h2 = torch.max(torch.min(dist, dim=1)[0])
        return torch.max(torch.tensor([h1, h2]))

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)/len(kernel_val)

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        # print(loss)
        return loss


class OT(nn.Module):
    def __init__(self, k: int,):
        super().__init__()
        self.k = k

    def forward(self, source_feat, target_feat, source_label, target_label):
        s_batch, num, dim, h, w = source_feat.size()
        t_batch, num, dim, h, w = target_feat.size()
        # t_feats = torch.zeros_like(target_feat)
        trans_mat = torch.zeros([num, dim, dim]).cuda()

        for k in range(num):
            print(k)
            s_mat = []
            t_mat = []
            samples_s = []
            samples_t = []
            # sample from all pixel in each image (by kmeans)
            for i in range(s_batch):
                feat_source = source_feat[i, k].view(dim, h*w).permute(1, 0)
                _, centres = kmeans(feat_source, self.k, distance='euclidean', device=torch.device('cuda:0'))
                samples_s.append(centres)
            for j in range(t_batch):
                feat_target = target_feat[j, k].view(dim, h*w).permute(1, 0)
                _, centres = kmeans(feat_target, self.k, distance='euclidean', device=torch.device('cuda:0'))
                samples_t.append(centres)

            for i in range(s_batch):
                for j in range(t_batch):
                    sample_s, sample_t = samples_s[i], samples_t[j]
                    dist = mat_dist(sample_s, sample_t)
                    # _, _, y = lapjv(dist.cpu().numpy())
                    _, y = torch.min(dist, dim=1)

                    sample_t = sample_t[y]
                    s_mat.append(sample_s)
                    t_mat.append(sample_t)
            s_mat = torch.cat([s for s in s_mat])
            t_mat = torch.cat([t for t in t_mat])
            # trans_mat[k] = torch.div(s_mat.t(), t_mat.t())
            # trans_mat[k] = s_mat.t().mm(torch.linalg.inv(t_mat.mm(t_mat.t())+1e-4)).mm(t_mat)
            trans_mat[k] = s_mat.t().mm(torch.pinverse(t_mat.t()))
            # for j in range(t_batch):
            #     # print(trans_mat[k].size())
            #     # print(target_feat[j, k].size())
            #     t_feats[j, k] = torch.matmul(target_feat[j, k].permute(1, 2, 0), trans_mat[k].t()).permute(2, 0, 1)
            #     # t_feats[j, k] = torch.matmul(trans_mat[k], target_feat[j, k])

        return trans_mat

#
# def vec_dist(vec1, vec2):
#     return np.linalg.norm(vec1 - vec2)

# def kmeans(feats, k):
#     # initial k centres
#     rands = [int(np.random.random() * feats.shape[0]) for _ in range(k)]
#     centres = [feats[rands[i]] for i in range(k)]
#     classes = [[] for _ in range(k)]
#
#     for i in range(0, feats.shape[0] - 1, 100):
#         dists = [vec_dist(feats[i], centre) for centre in centres]
#         for j in range(k):
#             if min(dists) == vec_dist(feats[i], centres[j]):
#                 classes[j].append(feats[i])
#                 centres[j] = np.mean(classes[j], axis=0)
#                 break
#     print(centres)
#     centres = torch.cat([torch.Tensor(c) for c in centres])
#     print(centres)
#     return centres