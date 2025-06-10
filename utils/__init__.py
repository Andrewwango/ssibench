import deepinv as dinv
from .moc import MOConsistencyLoss
from .splitting_consistency import SplittingConsistencyLoss
from .custom_loss import YourOwnLoss

class AdjointMCLoss(dinv.loss.MCLoss):
    def forward(self, y, x_net, physics, **kwargs):
        return self.metric(physics.A_adjoint(physics.A(x_net)), physics.A_adjoint(y))