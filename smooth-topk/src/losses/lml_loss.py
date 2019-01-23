import torch
from torch import nn

import sys
sys.path.append('../../') # TODO


from lml import LML

class LMLLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0):
        super(LMLLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

    def forward(self, x, y):
        n_batch = x.shape[0]

        p = LML(N=self.k, eps=1e-4)(x/self.tau)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8)
        return losses.mean()
