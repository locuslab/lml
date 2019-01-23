import torch
from torch import nn

class EntrLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0):
        super(EntrLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

    def forward(self, x, y):
        n_batch = x.shape[0]

        x = x/self.tau
        x_sorted, I = x.sort(dim=1, descending=True)
        x_sorted_last = x_sorted[:,self.k:]
        I_last = I[:,self.k:]

        fy = x.gather(1, y.unsqueeze(1))
        J = (I_last != y.unsqueeze(1)).type_as(x)

        # Could potentially be improved numerically by using
        # \log\sum\exp{x_} = c + \log\sum\exp{x_-c}
        safe_z = torch.clamp(x_sorted_last-fy, max=80)
        losses = torch.log(1.+torch.sum(safe_z.exp()*J, dim=1))

        return losses.mean()
