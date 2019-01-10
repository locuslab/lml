import torch
from torch import nn

import sys
sys.path.append('../../') # TODO


from lml import LML

class LMLLoss(nn.Module):
    def __init__(self, n_classes, k=5):
        super(LMLLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k

    def forward(self, x, y):
        n_batch = x.shape[0]

        p = LML(N=self.k, eps=1e-4)(x)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)))
        return losses.mean()
