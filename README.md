# SSIBench: a Self-Supervised Imaging Benchmark

> SSIBench is a modular benchmark for learning to solve imaging inverse problems without ground truth, applied to accelerated MRI reconstruction.

[Andrew Wang](https://andrewwango.github.io), [Steven McDonagh](https://smcdonagh.github.io/), [Mike Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)

[![arXiv](https://img.shields.io/badge/arXiv-2502.14009-b31b1b.svg)](https://arxiv.org/abs/2502.14009)
[![Code](https://img.shields.io/badge/GitHub-Code-blue.svg)](https://github.com/ssibench/ssibench)
[![Benchmark](https://img.shields.io/badge/Web-Benchmark-ff69b4.svg)](https://ssibench.github.io/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lSoR1vX-imvnJKcTvGS951ISlZjVRQfE?usp=sharing)

---

![](img/ssibench.svg)

Skip to...

1. [Overview](#overview)
2. [How to...](#how-to)  
    a. [...use the benchmark](#how-to-use-the-benchmark)  
    b. [...contribute a method](#how-to-contribute-a-method)  
    c. [...use a custom dataset](#how-to-use-a-custom-dataset), [model](#how-to-use-a-custom-model), [forward operator/acquisition strategy](#how-to-use-a-custom-forward-operatoracquisition-strategy), [metric](#how-to-use-a-custom-metric)  
3. [Benchmark results](#benchmark-results)
4. [Training script step-by-step](#training-script-step-by-step)

---

## Overview

SSIBench is a modular benchmark for learning to solve imaging inverse problems without ground truth, applied to accelerated MRI reconstruction. We contribute:

1. A comprehensive review of state-of-the-art self-supervised feedforward methods for inverse problems;
2. Well-documented implementations of all benchmarked methods in the open-source [DeepInverse](https://deepinv.github.io/) library, and a modular [benchmark site](https://ssibench.github.io/) enabling ML researchers to evaluate new methods or on custom setups and datasets;
3. Benchmarking experiments on MRI, across multiple realistic, general scenarios;
4. A new method, multi-operator equivariant imaging (MO-EI).

---

## How to…

### How to use the benchmark

First setup your environment:

1. Create a python environment:
```bash
python -m venv venv
source venv/Scripts/activate
```
2. Clone the benchmark repo:
```bash
git clone https://github.com/ssibench/ssibench.git
```
3. Install [DeepInverse](https://deepinv.github.io/)
```bash
pip install deepinv
```

Then run [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) for your chosen loss, where `--loss` is one of `mc`, ...:

```bash
python train.py --loss ...
```

To evaluate, use the same script [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) with 0 epochs and loading a checkpoint. We provide one pretrained model for quick eval for TODO

```bash
python train.py --epochs 0 --ckpt "...pt"
```

Notation: in our benchmark, we compare the `loss` functions $\mathcal{L}(\ldots)$, while keeping constant the `model` $f_\theta$, forward operator `physics` $A$, and data $y$.

### How to contribute a method

1. Add the code for your loss in the format:
```python
class YourOwnLoss(deepinv.loss.Loss):
    def forward(
        self, 
        x_net: torch.Tensor,    # Reconstruction i.e. model output
        y: torch.Tensor,        # Measurement data e.g. k-space in MRI
        x: torch.Tensor = None, # Ground truth, must be unused!
        model: deepinv.models.Reconstructor, # Reconstruction model $f_\theta$
        physics: deepinv.physics.Physics,    # Forward operator physics $A$
        **kwargs
    ):
        loss_calc = ...
        return loss_calc
```
2. Add your loss function as an option in [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) (hint: search _"Add your custom loss here!"_)
3. Open a [GitHub pull request](https://github.com/ssibench/ssibench/pulls) to contribute your loss! (hint: [how to open a PR in GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request))


### How to use a custom dataset

Our modular benchmark lets you easily train and evaluate the benchmarked methods on your own setup.

1. The custom dataset should have the form (see [DeepInverse docs](https://deepinv.github.io/deepinv/api/stubs/deepinv.Trainer.html#deepinv.Trainer:~:text=of%20the%20following-,options,-%3A) for details):
```python
class YourOwnDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx: int):
        ...
        # y = measurement data
        # params = dict of physics data-dependent parameters, e.g. acceleration mask in MRI
        return     x,     y, params # If ground truth x provided for evaluation
        return torch.nan, y, params # If ground truth does not exist
```
2. Replace `dataset = ...` in [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

### How to use a custom model

1. The custom model should have the form (see [DeepInverse guide](https://deepinv.github.io/deepinv/user_guide/reconstruction/introduction.html) for details):
```python
class YourOwnModel(deepinv.models.Reconstructor):
    def forward(
        self, 
        y: torch.Tensor,
        physics: deepinv.physics.Physics,
        **kwargs
    ):
        x_net = ...
        return x_net
```
2. Replace `model = ...` in [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

### How to use a custom forward operator/acquisition strategy

1. To use an alternative physics, you can use a different off-the-shelf [DeepInverse physics](https://deepinv.github.io/deepinv/user_guide/physics/physics.html) or a custom one of the form (see [DeepInverse guide](https://deepinv.github.io/deepinv/user_guide/physics/defining.html) on creating custom physics):
```python
class YourOwnPhysics(deepinv.physics.Physics):
    def A(self, x: torch.Tensor, **kwargs):
        y = ...
        return y
    
    def A_adjoint(self, y: torch.Tensor, **kwargs):
        x_hat = ...
        return x_hat
```
2. Replace `physics = ...` [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

### How to use a custom metric

1. The custom metric should have the form (see [DeepInverse docs](https://deepinv.github.io/deepinv/user_guide/training/metric.html) for details):
```python
class YourOwnMetric(dinv.loss.metric.Metric):
    def metric(
        self, 
        x_net: torch.Tensor, # Reconstruction i.e. model output
        x: torch.Tensor,     # Ground-truth for evaluation
    ):
        return ...
```
2. Replace `metric = ...` in [`train.py`](https://github.com/ssibench/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

---

## Benchmark results

TODO

---

## Training script step-by-step

step by step python …