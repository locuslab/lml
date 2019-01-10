#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import itertools
import time

import os

from torch.autograd import Variable

import sys
sys.path.insert(1, '.')

from losses.polynomial.sp import LogSumExp
from losses.utils import split
from tests.th_ref import log_sum_exp_k

sys.path.append('../../')
from lml import LML

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def sum_k_pyref(x, k):
    exp = torch.exp(x.data.cpu().numpy())
    n_classes = x.shape[1]
    res = 1e-10 * np.ones(len(x))
    for indices in itertools.combinations(range(n_classes), k):
        res += np.product(exp[:, indices], axis=1)
    return res


def esf_py(x, k, buffer):
    xx = torch.exp(x)
    n = x.size(1)

    # use buffer below
    buffer.zero_()
    res = Variable(buffer)
    res[:, :-1, 0] = 1
    res[:, 0, 1] = xx[:, 0]

    xx = xx.unsqueeze(2)

    for i in range(1, n):
        m = max(1, i + k - n)
        M = min(i, k) + 1
        res[:, i, m:M] = res[:, i - 1, m:M] + \
            xx[:, i] * res[:, i - 1, m - 1: M - 1]

    return torch.log(res[:, -1, k - 1]), torch.log(res[:, -1, k])


parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--n_classes', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--n_trials', type=int, default=50)
parser.add_argument('--no-cuda', action='store_true')

fname = '../xp/perf_results.txt'
if not os.path.exists(fname):
    f = open(fname, 'w')
    f.write('k,n_classes,batch_size,n_trials,tau,cuda,alg,forward,mean,std\n')
else:
    f = open(fname, 'a')

args = parser.parse_args()
k = args.k
CUDA = (not args.no_cuda) and torch.cuda.is_available()
tau = args.tau
batch_size = args.batch_size
n_classes = args.n_classes
n_trials = args.n_trials

print("=" * 70)
print("CONFIGURATION")
print("=" * 70)
print('-' * 70)
print('k: \t\t{}'.format(k))
print('C: \t\t{}'.format(n_classes))
print('Batch size: \t{}'.format(batch_size))
print('Tau: \t\t{}'.format(tau))
print('Cuda: \t\t{}'.format(CUDA))
print('n_trials: \t\t{}'.format(n_trials))
print('-' * 70)

torch.manual_seed(1234)

scores = Variable(torch.randn(batch_size, n_classes))
target = torch.from_numpy(np.random.randint(n_classes, size=batch_size))
labels = torch.from_numpy(np.arange(n_classes))

if CUDA:
    target = target.cuda()
    labels = labels.cuda()
    scores = scores.cuda()

x_1, x_2 = split(scores, Variable(target), labels)
x_1.div_(k * tau)
x_2.div_(k * tau)


def timing_fun(fun, x, k, verbosity, double=False, n_trials=50,
               forward=1, use_buffer=False):
    times = []
    for _ in range(n_trials):
        if double:
            x = x.double()
        x = Variable(x.data.clone(), requires_grad=not forward)
        if use_buffer:
            buffer = x.data.new(x.size(0), x.size(1), k + 1)
        if CUDA:
            torch.cuda.synchronize()
        if forward:
            clock = -time.time()
        if use_buffer:
            skm1, sk = fun(x, k, buffer)
        else:
            skm1, sk = fun(x, k)
        if not forward:
            if CUDA:
                torch.cuda.synchronize()
            clock = -time.time()
            (skm1 + sk).sum().backward()
        if CUDA:
            torch.cuda.synchronize()
        clock += time.time()

        times.append(clock)

        if verbosity:
            print(torch.stack((skm1.data, sk.data), dim=1).sum())

    return np.mean(times), np.std(times)


def speed(verbosity=1, forward=1):

    print("=" * 70)
    print("SPEED")
    print("=" * 70)

    print('-' * 70)
    if forward:
        print('FORWARD')
    else:
        print('BACKWARD')
    print('-' * 70)

    if not forward:
        mean, stdev = timing_fun(
            log_sum_exp_k, x_1, k, verbosity,
            double=False, forward=forward,
            n_trials=n_trials
        )
        print("Divide-and-conquer AD: {:.3e}s +/- {:.3e}s / mini-batch".format(mean, stdev))
        f.write(','.join(map(str, [
            k, n_classes, batch_size, n_trials, tau, CUDA, 'dac_ad', forward, mean, stdev
        ])) + '\n')

    if forward:
        mean, stdev = timing_fun(
            esf_py, x_1, k, verbosity,
            double=False, forward=forward,
            use_buffer=True, n_trials=n_trials
        )
        print("Summation Algorithm: {:.3e}s +/- {:.3e}s / mini-batch".format(mean, stdev))
        f.write(','.join(map(str, [
            k, n_classes, batch_size, n_trials, tau, CUDA, 'sum', forward, mean, stdev
        ])) + '\n')

    mean, stdev = timing_fun(
        lambda x, y: LogSumExp(k)(x), x_1, k, verbosity,
        double=False, forward=forward, n_trials=n_trials,
    )
    print("Divide-and-conquer MD: {:.3e}s +/- {:.3e}s / mini-batch".format(mean, stdev))
    f.write(','.join(map(str, [
        k, n_classes, batch_size, n_trials, tau, CUDA, 'dac_md', forward, mean, stdev
    ])) + '\n')

    mean, stdev = prof_lml(
        x_1, k, verbosity, double=False, forward=forward, n_trials=n_trials
    )
    f.write(','.join(map(str, [
        k, n_classes, batch_size, n_trials, None, CUDA, 'lml', forward, mean, stdev
    ])) + '\n')
    print("LML: {:.3e}s +/- {:.3e}s / mini-batch".format(mean, stdev))

    mean, stdev = prof_entr(
        x_1, k, verbosity, double=False, forward=forward, n_trials=n_trials
    )
    f.write(','.join(map(str, [
        k, n_classes, batch_size, n_trials, None, CUDA, 'entr', forward, mean, stdev
    ])) + '\n')
    print("entr: {:.3e}s +/- {:.3e}s / mini-batch".format(mean, stdev))


def prof_lml(x, k, verbosity, double=False, n_trials=50, forward=1):
    times = []
    for _ in range(n_trials):
        if double:
            x = x.double()
        x = Variable(x.data.clone(), requires_grad=not forward)

        if CUDA:
            torch.cuda.synchronize()
        if forward:
            clock = -time.time()

        p = LML(N=k, eps=1e-4)(x)

        if not forward:
            if CUDA:
                torch.cuda.synchronize()
            clock = -time.time()
            p.sum().backward()
        if CUDA:
            torch.cuda.synchronize()
        clock += time.time()

        times.append(clock)

        if verbosity:
            print(torch.stack((skm1.data, sk.data), dim=1).sum())

    return np.mean(times), np.std(times)


def prof_entr(x, k, verbosity, double=False, n_trials=50, forward=1):
    times = []
    for _ in range(n_trials):
        if double:
            x = x.double()
        x = Variable(x.data.clone(), requires_grad=not forward)

        if CUDA:
            torch.cuda.synchronize()
        if forward:
            clock = -time.time()

        p, _ = x.sort(dim=1, descending=True)

        if not forward:
            if CUDA:
                torch.cuda.synchronize()
            clock = -time.time()
            p.sum().backward()
        if CUDA:
            torch.cuda.synchronize()
        clock += time.time()

        times.append(clock)

        if verbosity:
            print(torch.stack((skm1.data, sk.data), dim=1).sum())

    return np.mean(times), np.std(times)



def run_fun(fun, x, k, double=False, use_buffer=False):

    if double:
        x = x.double()
    x = Variable(x.data.clone())

    if use_buffer:
        buffer = x.data.new(x.size(0), x.size(1), k + 1)
    if use_buffer:
        skm1, sk = fun(x, k, buffer)
    else:
        skm1, sk = fun(x, k)

    return skm1.data.cpu().numpy().sum() + sk.data.cpu().numpy().sum()


def stability():

    print("=" * 70)
    print("STABILITY")
    print("=" * 70)

    print('\n(Test successful if the number is not inf / nan)\n')

    res = run_fun(esf_py, x_1, k, double=False, use_buffer=True)
    print("Summation Algorithm (S): \t{}".format(res))

    res = run_fun(esf_py, x_1, k, double=True, use_buffer=True)
    print("Summation Algorithm (D): \t{}".format(res))

    res = run_fun(lambda x, y: LogSumExp(k)(x), x_1, k, double=False)
    print("Divide-and-conquer (S): \t{}".format(res))

    res = run_fun(lambda x, y: LogSumExp(k)(x), x_1, k, double=True)
    print("Divide-and-conquer (D): \t{}".format(res))


speed(verbosity=0, forward=1)
speed(verbosity=0, forward=0)

# stability()
