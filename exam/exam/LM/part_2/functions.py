import torch
import torch.nn as nn
import torch.optim as optim

import math
from utils import *
from functions import *

class VariationalDropout(nn.Module):
    # param dropout - dropout probability
    def __init__(self, dropout, mask):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
        self.mask = mask

    def forward(self, x):
        if not self.training:
            return x
        else:
            return x * self.mask
        

class NonMonotonicTriggeredAvSGD(optim.Optimizer):
    # Adjust the learning rate based on the training progress
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(NonMonotonicTriggeredAvSGD, self).__init__(params, defaults)
        self.losses = []
        self.min_loss = math.inf
        self.non_monotonic_steps = 0
        self.trigger = 3

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        if loss is not None:
            self.losses.append(loss)
            if loss < self.min_loss:
                self.min_loss = loss
                self.non_monotonic_steps = 0
            else:
                self.non_monotonic_steps += 1
                if self.non_monotonic_steps >= self.trigger:
                    self.non_monotonic_steps = 0
                    group['lr'] *= 0.1  # Reduce learning rate
        return loss