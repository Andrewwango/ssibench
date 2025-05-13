# SSIBench: Benchmarking Self-Supervised Learning Methods for Accelerated MRI Reconstruction

> SSIBench is a modular benchmark for learning to solve imaging inverse problems without ground truth, applied to accelerated MRI reconstruction.

[Andrew Wang](https://andrewwango.github.io), [Steven McDonagh](https://smcdonagh.github.io/), [Mike Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)

[![arXiv](https://img.shields.io/badge/arXiv-2502.14009-b31b1b.svg)](https://arxiv.org/abs/2502.14009)
[![Code](https://img.shields.io/badge/GitHub-Code-blue.svg)](https://github.com/Andrewwango/ssibench)
[![Benchmark](https://img.shields.io/badge/Web-Benchmark-ff69b4.svg)](https://andrewwango.github.io/ssibench)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15CFQ7Lp9Xk-J3ltJ6kWHIs5tk8Z4VBYD?usp=sharing)

---

![](img/ssibench.svg)

Skip to...

1. [Overview](#overview)
2. [How to...](#how-to)  
    a. [...use the benchmark](#how-to-use-the-benchmark)  
    b. [...contribute a method](#how-to-contribute-a-method)  
    c. [...use a custom dataset](#how-to-use-a-custom-dataset), [model](#how-to-use-a-custom-model), [forward operator/acquisition strategy](#how-to-use-a-custom-forward-operatoracquisition-strategy), [metric](#how-to-use-a-custom-metric)  
3. [Live leaderboard](#live-leaderboard)
4. [Training script step-by-step](#training-script-step-by-step)
5. [Dataset preparation instructions](#dataset-preparation-instructions)

---

## Overview

SSIBench is a modular benchmark for learning to solve imaging inverse problems without ground truth, applied to accelerated MRI reconstruction. We contribute:

1. A comprehensive review of state-of-the-art self-supervised feedforward methods for inverse problems;
2. Well-documented implementations of all benchmarked methods in the open-source [DeepInverse](https://deepinv.github.io/) library, and a modular [benchmark site](https://andrewwango.github.io/ssibench) enabling ML researchers to evaluate new methods or on custom setups and datasets;
3. Benchmarking experiments on MRI, on a standardised setup across multiple realistic, general scenarios;
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
git clone https://github.com/Andrewwango/ssibench.git
```
3. Install [DeepInverse](https://deepinv.github.io/)
```bash
pip install deepinv
```
4. Prepare your [fastMRI](https://fastmri.med.nyu.edu/) data using the [below instructions](#dataset-preparation-instructions).

Then run [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) for your chosen loss, where `--loss` is the loss function (`mc`, `ei` etc.), and `--physics` is the physics (see [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) for options):

```bash
python train.py --loss ... --physics ...
```

To evaluate, use the same script [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) with 0 epochs and loading a checkpoint. We provide one pretrained model for quick eval ([download here](https://huggingface.co/Andrewwango/ssibench/blob/main/demo_mo-ei.pth.tar)):

```bash
python train.py --epochs 0 --ckpt "demo_mo-ei.pth.tar"
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
        model: deepinv.models.Reconstructor, # Reconstruction model $f_\theta$
        physics: deepinv.physics.Physics,    # Forward operator physics $A$
        x: torch.Tensor = None, # Ground truth, must be unused!
        **kwargs
    ):
        loss_calc = ...
        return loss_calc
```
2. Add your loss function as an option in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) (hint: search _"Add your custom loss here!"_)
3. Benchmark your method by running [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) (hint: [_"How to use the benchmark"_](#how-to-use-the-benchmark)).
4. Submit your results by editing the [live leaderboard](#live-leaderboard).
5. Open a [GitHub pull request](https://github.com/Andrewwango/ssibench/pulls) to contribute your loss! (hint: [see example here](https://github.com/Andrewwango/ssibench/pull/1); hint: [how to open a PR in GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request))


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
2. Replace `dataset = ...` in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

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
2. Replace `model = ...` in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

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
2. Replace `physics = ...` [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

### How to use a custom metric

1. The custom metric should have the form (see [DeepInverse docs](https://deepinv.github.io/deepinv/user_guide/training/metric.html) for details):
```python
class YourOwnMetric(deepinv.loss.metric.Metric):
    def metric(
        self, 
        x_net: torch.Tensor, # Reconstruction i.e. model output
        x: torch.Tensor,     # Ground-truth for evaluation
    ):
        return ...
```
2. Replace `metrics = ...` in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) with your own, then train/evaluate using the script as in [How to use the benchmark](#how-to-use-the-benchmark).

---

## Live leaderboard

We provide a live leaderboard for each experimental scenario described in [the paper](https://arxiv.org/abs/2502.14009). Got a new method? [Contribute it](#how-to-contribute-a-method) to the leaderboard!

<details><summary>Scenario 1 (single-coil)</summary>

| # |     **Loss**    | **PSNR**| **SSIM**|
| --- | --- | --- | --- |
| 1 | UAIR | 14.00 | .3715 |
| 2 | Adversarial | 18.52 | .4732 |
| 3 | MC  | 27.66 | .7861 |
| 4 | Zero-filled | 27.67 | .7862 |
| 5 | VORTEX | 27.75 | .7898 |
| 6 | SSDU | 27.98 | .7485 |
| 7 | Noise2Inverse | 28.42 | .7853 |
| 8 | Weighted-SSDU | 29.93 | .8355 |
| 9 | EI  | 30.26 | .8523 |
| 10 | MOI | 30.29 | .8651 |
| 11 | MOC-SSDU | 30.42 | .8198 |
| 12 | SSDU-Consistency | 30.81 | .8495 |
| 13 | MO-EI (ours) | 32.14 | .8846 |
| 14 | (Supervised) | 33.15 | .9032 |

![](img/results_brain.png)

</details>

<details><summary>Scenario 2 (noisy)</summary>

| # |     **Loss**    | **PSNR**| **SSIM**|
| --- | --- | --- | --- |
| 1 | Zero-filled | 24.34 | .4428 |
| 2 | (Non-robust) Weighted-SSDU | 25.91 | .5477 |
| 3 | (Non-robust) MO-EI | 26.12 | .6002 |
| 4 | ENSURE | 26.29 | .5856 |
| 5 | Robust-SSDU | 27.42 | .6159 |
| 6 | Noise2Recon-SSDU | 27.84 | .7661 |
| 7 | Robust-EI | 29.07 | .8227 |
| 8 | Robust-MO-EI | 29.72 | .8409 |
| 9 | (Supervised) | 30.19 | .8411 |

![](img/results_brain_noisy.png)

</details>

<details><summary>Scenario 3 (single-operator)</summary>

| # |     **Loss**    | **PSNR**| **SSIM**|
| --- |----------------|-----|-----|
| 1 |      UAIR      |18.44|.5388|
| 2 |      SSDU      |21.89|.6288|
| 3 |  Noise2Inverse |24.63|.6559|
| 4 |   Adversarial  |26.53|.7013|
| 5 |    MOC-SSDU    |27.85|.7717|
| 6 |   Zero-filled  |28.02|.7900|
| 7 |       MC       |28.02|.7900|
| 8 |     VORTEX     |28.07|.7916|
| 9 |  Weighted-SSDU |30.14|.8454|
| 10 |SSDU-Consistency|31.05|.8614|
| 11 |  MO-EI (ours)  |31.11|.8713|
| 12 |       MOI      |31.60|.8789|
| 13 |       EI       |31.99|.8806|
| 14 |  (Supervised)  |34.03|.9040|

![](img/results_brain_single.png)

</details>

<details><summary>Scenario 4 (multi-coil)</summary>

| # |     **Loss**    | **PSNR**| **SSIM**|
| --- |----------------|-----|-----|
| 1 |      UAIR      |15.26|.3453|
| 2 |   Adversarial  |17.47|.6464|
| 3 |     VORTEX     |23.59|.5846|
| 4 |   Zero-filled  |27.82|.7988|
| 5 |       MC       |28.96|.8271|
| 6 |  Noise2Inverse |30.93|.8589|
| 7 |       MOI      |31.37|.8810|
| 8 |      SSDU      |31.47|.8705|
| 9 |  MO-EI (ours)  |31.56|.8836|
| 10 |       EI       |31.66|.8769|
| 11 |    MOC-SSDU    |31.80|.8761|
| 12 |SSDU-Consistency|32.30|.8949|
| 13 |  Weighted-SSDU |33.03|.8991|
| 14 |  (Supervised)  |33.89|.9147|

![](img/results_brain_multicoil.png)

</details>

---

## Training script step-by-step
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15CFQ7Lp9Xk-J3ltJ6kWHIs5tk8Z4VBYD?usp=sharing)

The training script makes extensive use of modular training framework provided by [DeepInverse](https://deepinv.github.io).

```{python}
import deepinv as dinv
import torch
```

Define training parameters:

```{python}
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)
acceleration = 6
batch_size = 4
lr = 1e-3
img_size = (320, 320)

class args: # Command line args from train.py
    physics = "mri"
    epochs = 0
    loss = "mc"
    ckpt = None
```

Define MRI physics $A$ and mask generator $M$ according to scenario

```{python}
physics_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=acceleration, rng=rng, device=device)
physics = dinv.physics.MRI(img_size=img_size, device=device)

match args.physics:
    case "noisy":
        sigma = 0.1
        physics.noise_model = dinv.physics.GaussianNoise(sigma, rng=rng)
    case "multicoil":
        physics = dinv.physics.MultiCoilMRI(img_size=img_size, coil_maps=4, device=device)
    case "single":
        physics.update(**physics_generator.step())
```

Define model $f_\theta$

```{python}
denoiser = dinv.models.UNet(2, 2, scales=4, batch_norm=False)
model = dinv.models.MoDL(denoiser=denoiser, num_iter=3).to(device)
```

Define dataset

```{python}
dataset = dinv.datasets.SimpleFastMRISliceDataset("data", file_name="fastmri_brain_singlecoil.pt")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2), generator=rng_cpu)
```

Simulate and save random measurements
```{python}
dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    physics_generator=physics_generator if args.physics != "single" else None,
    save_physics_generator_params=True,
    overwrite_existing=False,
    device=device,
    save_dir="data",
    batch_size=1,
    dataset_filename="dataset_" + args.physics
)

train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
test_dataset  = dinv.datasets.HDF5Dataset(dataset_path, split="test",  load_physics_generator_params=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=rng_cpu)
test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size)
```

Define loss function (see [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) for all options)

```{python}
match args.loss:
    case "mc":
        loss = dinv.loss.MCLoss()

    case "...":
        # Add your custom loss here!
        pass
```

Define metrics

```{python}
metrics = [
    dinv.metric.PSNR(complex_abs=True),
    dinv.metric.SSIM(complex_abs=True)
]
```
Define trainer

```{python}
trainer = dinv.Trainer(
    model = model,
    physics = physics,
    optimizer = torch.optim.Adam(model.parameters(), lr=lr),
    train_dataloader = train_dataloader,
    eval_dataloader = test_dataloader,
    epochs = args.epochs,
    losses = loss,
    metrics = metrics,
    device = device,
    ckpt_pretrained=args.ckpt,
)
```

```{python}
#| code-fold: true
#| code-summary: Define additional adversarial trainer (if needed)

if args.loss in ("uair", "adversarial"):
    trainer = dinv.training.AdversarialTrainer(
        model = model,
        physics = physics,
        optimizer = dinv.training.AdversarialOptimizer(
            torch.optim.Adam(model.parameters(), lr=lr), 
            torch.optim.Adam(discrim.parameters(), lr=lr)
        ),
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = args.epochs,
        losses = loss,
        metrics = metrics,
        device = device,
        ckpt_pretrained=args.ckpt,
    )

    trainer.D = discrim
    trainer.losses_d = loss_d
```

Train or evaluate!

```{python}
trainer.train()

print(trainer.test(test_dataloader))
```

# Dataset preparation instructions

To prepare the fastMRI dataset `fastmri_brain_singlecoil.pt` used in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py) for the benchmark experiments, we make use of the [fastMRI wrapper in DeepInverse](https://deepinv.github.io/deepinv/api/stubs/deepinv.datasets.FastMRISliceDataset.html).

1. Download [fastMRI](https://fastmri.med.nyu.edu/) brain dataset batch 0: `brain_multicoil_train_batch_0` (~98.5 GB)
2. Generate an efficient dataset of the middle slices (note that this is deterministic, and the random masks & noise are simulated in [`train.py`](https://github.com/Andrewwango/ssibench/blob/main/train.py)):

```{python}
dataset = deepinv.datasets.FastMRISliceDataset(
    "/path/to/fastmri/brain/multicoil_train", 
    slice_index="middle"
)

dataset.save_simple_dataset("data/fastmri_brain_singlecoil.pt")
```