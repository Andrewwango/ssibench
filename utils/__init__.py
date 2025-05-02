import deepinv as dinv
from .ensure import ENSURELoss

class AdjointMCLoss(dinv.loss.MCLoss):
    def forward(self, y, x_net, physics, **kwargs):
        return self.metric(physics.A_adjoint(physics.A(x_net)), physics.A_adjoint(y))