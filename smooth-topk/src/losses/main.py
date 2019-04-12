import torch.nn as nn
from losses.svm import SmoothSVM
from losses.lml_loss import LMLLoss
from losses.ml import MLLoss
from losses.entr import EntrLoss

def get_loss(xp, args):
    if args.loss == "svm":
        print("Using SVM loss")
        loss = SmoothSVM(n_classes=args.num_classes, k=args.topk, tau=args.tau, alpha=args.alpha)
    elif args.loss == 'ce':
        print("Using CE loss")
        loss = nn.CrossEntropyLoss()
        loss.tau = -1
    elif args.loss == 'lml':
        print("Using LML loss")
        loss = LMLLoss(n_classes=args.num_classes, k=args.topk, tau=args.tau)
    elif args.loss == 'ml':
        loss = MLLoss(n_classes=args.num_classes)
    elif args.loss == 'entr':
        print("Using truncated entr (Lapin) loss")
        loss = EntrLoss(n_classes=args.num_classes, k=args.topk, tau=args.tau)
    else:
        raise ValueError('Invalid choice of loss ({})'.format(args.loss))

    xp.Temperature.set_fun(lambda: loss.tau)

    return loss
