from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE


def feat_plot(source_feat, target_feat, source_label, target_label):
    pca2 = decomposition.PCA(2)
    tsne = TSNE(n_components=2, init='pca', random_state=501)

    s_batch, num, dim, h, w = source_feat.size()
    t_batch, num, dim, h, w = target_feat.size()
    ys = source_label[0].view(h*w)
    yt = target_label[0].view(h*w)

    for i in range(num):
        xs = source_feat[0, i].view(dim, h*w).permute(1, 0).cpu().detach().numpy()
        xt = target_feat[0, i].view(dim, h*w).permute(1, 0).cpu().detach().numpy()
        print('==>Start dimension reduce')
        xs_2d = tsne.fit_transform(xs)
        xt_2d = tsne.fit_transform(xt)
        # xs_2d = pca2.fit_transform(xs)
        # xt_2d = pca2.fit_transform(xt)
        print('==>Finish dimension reduce')

        xs_pos = xs_2d[ys == 1]
        xs_neg = xs_2d[ys == 0]
        xt_pos = xt_2d[yt == 1]
        xt_neg = xt_2d[yt == 0]

        plt.figure(1)
        N = 30
        plt.scatter(xs_pos[:N, 0], xs_pos[:N, 1], color='b')
        plt.scatter(xs_neg[:N, 0], xs_neg[:N, 1], color='r')
        plt.scatter(xt_pos[:N, 0], xt_pos[:N, 1], color='g')
        plt.scatter(xt_neg[:N, 0], xt_neg[:N, 1], color='coral')
        # plt.show()
        plt.savefig('fig/plot_{}.png'.format(i), bbox_inches='tight')


def prob_plot(prob, label, save_name):

    prob = prob.squeeze()
    label = label.squeeze()

    plt.figure(1)
    plt.hist()

    plt.show()
    plt.savefig(save_name, bbox_inches='tight')
