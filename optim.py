import torch
from torch.optim.optimizer import Optimizer, required


class Nesterov(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=required, momentum=required,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum is not required and lr < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(Nesterov, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Nesterov, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = (
                        torch.zeros_like(p.data))
                else:
                    buf = param_state['momentum_buffer']
                v = buf.mul_(momentum).add_(d_p.mul(group['lr']))
                p.data.add_(-1, v)

        return loss