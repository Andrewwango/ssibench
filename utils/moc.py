import torch
from deepinv.loss.moi import MOILoss

class MOConsistencyLoss(MOILoss):
    r"""
    Multi-operator consistency loss.

    Computes the following loss:

    .. math::

        \| y - A_s\inverse{A_g\hat{x},A_g} \|^2

    where :math:`\hat{x}=\inverse{y,A_s}` is a reconstructed signal (observed via operator :math:`A_s`) and
    :math:`A_g` is a forward operator sampled at random from a set :math:`\{A_g\}_{g=1}^{G}`.

    This is similar to `MOI <https://arxiv.org/abs/2201.12151>`_ :class:`deepinv.loss.MOILoss` but computes the consistency in the measurement domain rather than the image domain,
    and was proposed in `Zhang et al., CC-SSDU <https://ieeexplore.ieee.org/document/10635895>`_.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    The operators can be passed as a list of physics or as a single physics with a random physics generator.

    :param list[Physics], Physics physics: list of physics containing the :math:`G` different forward operators
            associated with the measurements, or single physics, or None. If single physics or None, physics generator must be used.
            If None, physics taken during ``forward``.
    :param PhysicsGenerator physics_generator: random physics generator that generates new params, if physics is not a list.
    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float weight: total weight of the loss
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param torch.Generator rng: torch randon number generator for randomly selecting from physics list. If using physics generator, rng is ignored.
    """

    class DummyMetric(torch.nn.Module):
        """Helper metric to return first argument of metric"""

        def forward(self, y1, y2):
            return y1

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Computes the multi-operator consistency loss.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor y: measurements.
        :param Physics physics: measurement physics.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """
        metric, weight = self.metric, self.weight
        self.metric, self.weight = self.DummyMetric(), 1

        x2 = super().forward(x_net, physics, model)
        
        self.metric, self.weight = metric, weight
        return self.weight * self.metric(y, physics.A(x2))    