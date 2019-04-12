import torch
from torch import nn

class MLLoss(nn.Module):
    def __init__(self, n_classes):
        super(MLLoss, self).__init__()
        self.n_classes = n_classes
        self.tau = 1.0

    def forward(self, x, y):
        n_batch = x.shape[0]
        y_onehot = torch.zeros(n_batch, self.n_classes).type_as(x)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        loss = nn.BCEWithLogitsLoss()(x, y_onehot)
        return loss
