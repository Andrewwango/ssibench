from copy import deepcopy
import torch
from deepinv.loss.measplit import SplittingLoss

class SplittingConsistencyLoss(SplittingLoss):
    r"""
    Splitting consistency loss.

    Computes the following loss:

    .. math::

        \ell(\forw{\hat{x}_1},y)+\ell(\forw{\hat{x}_2},\y) + \ell(\bar{A}\hat{x}_1,\bar{A}\hat{x}_2),\; \xhat_k = \inverse{M_k y,M_k A}
    
    where :math:`\ell` is any metric, which by default is the MSE, :math:`M_k` are two different splitting mask instances,
    and :math:`\bar{A} is the physics :math:`A` but with the complement of the mask :math:`I-M`.
    Note this loss only applicable when the physics has a mask i.e. :class:`deepinv.physics.MRI` or :class:`deepinv.physics.Inpainting`.

    This loss was proposed for MRI in `Hu et al. <https://link.springer.com/chapter/10.1007/978-3-030-87231-1_37>`_.
    Note we use the shared weights version of this loss when training a single model.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    :param deepinv.physics.generator.BernoulliSplittingMaskGenerator, None mask_generator: function to generate the mask.
    """
    def __init__(self, mask_generator = None, **kwargs):
        super().__init__(mask_generator=mask_generator, eval_split_input=False, eval_split_output=False, **kwargs)
    
    def forward(self, x_net, y, physics, model, **kwargs):     
        x_net2 = model(y, physics)

        assert not torch.all(x_net == x_net2) # should be new random masking

        physics_c = deepcopy(physics)
        physics_c.update(mask=1. - getattr(physics, "mask", 1.0))
        
        return self.metric(physics_c.A(x_net), physics_c.A(x_net2)) + self.metric(y, physics.A(x_net)) + self.metric(y, physics.A(x_net2))