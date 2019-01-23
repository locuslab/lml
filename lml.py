#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from semantic_version import Version
version = Version('.'.join(torch.__version__.split('.')[:3]))
old_torch = version < Version('0.4.0')

def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()

class LML(Function):
    def __init__(self, N, eps=1e-4, n_iter=100, branch=None):
        super().__init__()
        self.N = N
        self.eps = eps
        self.n_iter = n_iter
        self.branch = branch

    def forward(self, x):
        branch = self.branch
        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100

        single = x.ndimension() == 1
        orig_x = x
        if single:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= self.N:
            y = (1.-1e-5)*torch.ones(n_batch, nx).type_as(x)
            if single:
                y = y.squeeze(0)
            if old_torch:
                self.save_for_backward(orig_x)
                self.y = y
                self.nu = torch.Tensor()
            else:
                self.save_for_backward(orig_x, y, torch.Tensor())
            return y

        x_sorted, _ = torch.sort(x, dim=1, descending=True)

        # The sigmoid saturates the interval [-7, 7]
        nu_lower = -x_sorted[:,self.N-1] - 7.
        nu_upper = -x_sorted[:,self.N] + 7.

        ls = torch.linspace(0,1,branch).type_as(x)

        for i in range(self.n_iter):
            r = nu_upper-nu_lower
            I = r > self.eps
            n_update = I.sum()
            if n_update == 0:
                break

            Ix = I.unsqueeze(1).expand_as(x) if old_torch else I

            nus = r[I].unsqueeze(1)*ls + nu_lower[I].unsqueeze(1)
            _xs = x[Ix].view(n_update, 1, nx) + nus.unsqueeze(2)
            fs = torch.sigmoid(_xs).sum(dim=2) - self.N
            # assert torch.all(fs[:,0] < 0) and torch.all(fs[:,-1] > 0)

            i_lower = ((fs < 0).sum(dim=1) - 1).long()
            J = i_lower < 0
            if J.sum() > 0:
                print('LML Warning: An example has all positive iterates.')
                i_lower[J] = 0

            i_upper = i_lower + 1

            nu_lower[I] = nus.gather(1, i_lower.unsqueeze(1)).squeeze()
            nu_upper[I] = nus.gather(1, i_upper.unsqueeze(1)).squeeze()

            if J.sum() > 0:
                nu_lower[J] -= 7.

        if np.any(I.cpu().numpy()):
            print('LML Warning: Did not converge.')

        nu = nu_lower + r/2.
        y = torch.sigmoid(x+nu.unsqueeze(1))
        if single:
            y = y.squeeze(0)

        if old_torch:
            # Storing these in the object may cause memory leaks.
            self.save_for_backward(orig_x)
            self.y = y
            self.nu = nu
        else:
            self.save_for_backward(orig_x, y, nu)
        return y

    def backward(self, grad_output):
        if old_torch:
            x, = self.saved_tensors
            y = self.y
            nu = self.nu
        else:
            x, y, nu = self.saved_tensors

        single = x.ndimension() == 1
        if single:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        assert x.ndimension() == 2
        assert y.ndimension() == 2
        assert grad_output.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= self.N:
            dx = torch.zeros_like(x)
            if single:
                dx = dx.squeeze()
            return dx

        Hinv = 1./(1./y + 1./(1.-y))
        dnu = bdot(Hinv, grad_output)/Hinv.sum(dim=1)
        dx = -Hinv*(-grad_output+dnu.unsqueeze(1))

        if single:
            dx = dx.squeeze()

        return dx

if __name__ == '__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)

    m = 10
    n = 2

    npr.seed(0)
    x = npr.random(m)

    import cvxpy as cp
    import numdifftools as nd

    y = cp.Variable(m)
    obj = cp.Minimize(-x*y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.-y)))
    cons = [0 <= y, y <= 1, cp.sum(y) == n]
    prob = cp.Problem(obj, cons)
    prob.solve(cp.SCS, verbose=True)
    assert 'optimal' in prob.status
    y_cp = y.value

    x = Variable(torch.from_numpy(x), requires_grad=True)
    x = torch.stack([x,x])
    y = LML(N=n)(x)

    np.testing.assert_almost_equal(y[0].data.numpy(), y_cp, decimal=3)

    dy0, = grad(y[0,0], x)
    dy0 = dy0.squeeze()

    def f(x):
        x = Variable(torch.from_numpy(x).clone())
        y = LML(N=n)(x)
        return y.data.numpy()

    x = x.data[0].numpy().copy()
    df = nd.Jacobian(f)

    dy0_fd = df(x)[0]

    np.testing.assert_almost_equal(dy0[0].data.numpy(), dy0_fd, decimal=3)
