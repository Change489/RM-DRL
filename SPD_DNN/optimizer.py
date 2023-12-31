import torch
from torch.optim.optimizer import Optimizer

from PT import expm
from SPD_DNN import StiefelParameter, SPDParameter
from SPD_DNN.utils import orthogonal_projection, retraction


class StiefelMetaOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0
                if isinstance(p, StiefelParameter):
                    trans = orthogonal_projection(p.grad, p)

                    p.grad.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    riem = p @ ((p.grad + p.grad.transpose(-2, -1)) / 2) @ p
                    self.state[p] = p.clone()
                    p.fill_(0)
                    p.grad.fill_(0).add_(riem)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):

                    trans = retraction(p)

                    p.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    trans = expm(self.state[p], p)
                    p.fill_(0).add_(trans)

        return loss
