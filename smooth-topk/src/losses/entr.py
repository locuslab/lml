import torch
from torch import nn

class EntrLoss(nn.Module):
    def __init__(self, n_classes, k=5):
        super(EntrLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k

    def forward(self, x, y):
        n_batch = x.shape[0]

        x_sorted, I = x.sort(dim=1, descending=True)
        x_sorted_last = x_sorted[:,self.k:]
        I_last = I[:,self.k:]

        fy = x.gather(1, y.unsqueeze(1))
        J = (I_last != y.unsqueeze(1)).type_as(x)
        losses = torch.log(1.+torch.sum((x_sorted_last-fy).exp()*J, dim=1))
        return losses.mean()
