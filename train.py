import deepinv as dinv
import torch

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--epochs", type=int, default=0)  # Number of training epochs
parser.add_argument("--ckpt", type=str, default=None) # Pretrained model path
parser.add_argument("--loss", type=str, default="mc", choices=(
    "mc", "sup", 
    "ssdu", "noise2inverse", "weighted-ssdu", "ssdu-consistency",
    "adversarial", "uair",
    "vortex", "ei", "moi", "moc-ssdu", "mo-ei",
    "ensure", "noise2recon-ssdu", "robust-ssdu", "robust-ei", "robust-mo-ei",
    # Add your custom loss here!
))
parser.add_argument("--physics", type=str, default="mri", choices=(
    "mri",       # Scenario 1: Noiseless single-coil
    "noisy",     # Scenario 2: Noisy measurement single-coil
    "single",    # Scenario 3: Single-operator (i.e. single-mask) noiseless
    "multicoil", # Scenario 4: Multicoil noiseless
))
args = parser.parse_args()

# Define training parameters

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)
acceleration = 6
batch_size = 4
lr = 1e-3
img_size = (320, 320)

# Define MRI physics $A$ and mask generator $M$ according to scenario
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

# Define model $f_\theta$
denoiser = dinv.models.UNet(2, 2, scales=4, batch_norm=False)
model = dinv.models.MoDL(denoiser=denoiser, num_iter=3).to(device)

# Define dataset
dataset = dinv.datasets.SimpleFastMRISliceDataset("data", file_name="fastmri_brain_singlecoil.pt")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2), generator=rng_cpu)

# Simulate and save random measurements
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

# Define loss function
#
# Note that if the losses cannot be imported, they are likely still in the process of being merged.
# Check progress on DeepInverse https://github.com/deepinv/deepinv/pulls
# and if desired install DeepInverse from a PR branch.
from utils import *
match args.loss:
    case "mc":
        loss = dinv.loss.MCLoss() if args.physics != "multicoil" else AdjointMCLoss()

    case "sup":
        loss = dinv.loss.SupLoss()

    case "ssdu":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )

    case "noise2inverse":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=3
        )

    case "weighted-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=(1, *img_size), acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.mri.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)        
    
    case "moc-ssdu":
        loss = [
            MOConsistencyLoss(physics_generator=physics_generator),
            dinv.loss.SplittingLoss(
                mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng),
                eval_split_input=False
        )]

    case "ssdu-consistency":
        loss = SplittingConsistencyLoss(mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng))

    case "adversarial":
        discrim = dinv.models.gan.SkipConvDiscriminator(img_size, use_sigmoid=False).to(device)
        
        dataloader_factory = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator("cpu").manual_seed(42))
        physics_generator_factory = lambda: dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=acceleration, rng=torch.Generator(device).manual_seed(42), device=device)
        
        loss = dinv.loss.adversarial.MultiOperatorUnsupAdversarialGeneratorLoss(device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory, metric="A_adjoint")
        loss_d=dinv.loss.adversarial.MultiOperatorUnsupAdversarialDiscriminatorLoss(device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory, metric="A_adjoint")

    case "uair":
        discrim = dinv.models.gan.SkipConvDiscriminator(img_size, use_sigmoid=False).to(device)
        physics_generator_factory = lambda: dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=acceleration, rng=torch.Generator(device).manual_seed(42), device=device)
        loss = dinv.loss.adversarial.UAIRGeneratorLoss(device=device, physics_generator_factory=physics_generator_factory, metric_adv="A_adjoint" if args.physics == "multicoil" else None)
        loss_d=dinv.loss.adversarial.UAIRDiscriminatorLoss(device=device, physics_generator_factory=physics_generator_factory, metric_adv="A_adjoint" if args.physics == "multicoil" else None)
    
    case "vortex":
        loss = [
            dinv.loss.MCLoss() if args.physics != "multicoil" else AdjointMCLoss(), 
            dinv.loss.AugmentConsistencyLoss(dinv.transform.RandomPhaseError(scale=0.1, rng=rng, flatten_video_input=False) * dinv.transform.RandomNoise(rng=rng, flatten_video_input=False), rng=rng)
        ]

    case "ei":
        loss = [
            dinv.loss.MCLoss() if args.physics != "multicoil" else AdjointMCLoss(),
            dinv.loss.EILoss(transform=dinv.transform.Rotate())
        ]

    case "moi":
        loss = [
            dinv.loss.MCLoss() if args.physics != "multicoil" else AdjointMCLoss(),
            dinv.loss.MOILoss(physics_generator=physics_generator)
        ]

    case "mo-ei":
        loss = [
            dinv.loss.MCLoss() if args.physics != "multicoil" else AdjointMCLoss(),
            dinv.loss.MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator)
        ]

    case "ensure":
        loss = ENSURELoss(sigma=sigma, physics_generator=physics_generator)

    case "robust-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.mri.RobustSplittingLoss(mask_generator, physics_generator, dinv.physics.GaussianNoise(sigma=sigma, rng=rng))
    
    case "noise2recon-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = [
            dinv.loss.mri.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator),
            dinv.loss.AugmentConsistencyLoss(dinv.transform.RandomNoise(sigma=(sigma * 0.5, sigma * 2), rng=rng), dinv.transform.Identity(), no_grad=False)
        ]
    
    case "ddssl":
        loss = [
            dinv.loss.R2RLoss(noise_model=dinv.physics.GaussianNoise(sigma=sigma, rng=rng), alpha=0.5),
            dinv.loss.EILoss(dinv.transform.RandomNoise(sigma=sigma, rng=rng), apply_noise=False)
        ]

    case "robust-ei":
        loss = [dinv.loss.SureGaussianLoss(sigma=sigma), dinv.loss.EILoss(transform=dinv.transform.Rotate())]

    case "robust-mo-ei":
        loss = [dinv.loss.SureGaussianLoss(sigma=sigma), dinv.loss.MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator)]
    
    case "...":
        # Add your custom loss here!
        pass

# Define metrics
metrics = [
    dinv.metric.PSNR(complex_abs=True),
    dinv.metric.SSIM(complex_abs=True)
]

# Define trainer
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

# Define additional adversarial trainer
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

trainer.train()

print(trainer.test(test_dataloader))