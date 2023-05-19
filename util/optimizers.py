import torch

__all__ = ['init_optim']


def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, params), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, params), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))

