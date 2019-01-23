# The Limited Multi-Label Projection Layer

We provide the LML layer as a PyTorch module in `lml.py`
that can be imported and used as

```
from lml import LML

x = ...
y = LML(N=n)(x)
```

# Top-k Image Classification
In the `smooth-topk` directory, we have connected the LML layer to the
PyTorch experiments in the
[oval-group/smooth-topk](https://github.com/oval-group/smooth-topk)
repository.
We ran these experiments with PyTorch 1.0.

A single LML training run can be done from the `smooth-topk/src` directory with

```
./main.py --dataset cifar100 --model densenet40-40 --out-name /tmp/lml-cifar --loss lml --noise 0.0 --seed 0 --no-visdom
```

Coordinating all of the CIFAR-100 experiments can be done with
the `./scripts/cifar100_noise_*.sh` scripts.

# Neural Motifs: Scene Graph Generation

In the `neural-motifs` directory, we have connected the LML layer to the
PyTorch experiments in the
[rowanz/neural-motifs](https://github.com/rowanz/neural-motifs)
repository.
The `README` in this directory provides more details about
setting up and running the experiments.
The original code has not been updated to the latest version of
PyTorch and these experiments should be run with PyTorch 0.3.

A single LML training run can be done from the `neural-motifs` directory with

```
python3 models/train_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar -save_dir /tmp/lml-nm -nepoch 50 -use_bias --lml_topk 20
```

Coordinating all of the experiments can be done with
`/scripts/train_predcls.sh`.

# Licensing

Our LML layer in `lml.py` is licensed under the TODO license.
All other code in this repository remains under the
original licensing.
