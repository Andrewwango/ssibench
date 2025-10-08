import deepinv
import torch

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