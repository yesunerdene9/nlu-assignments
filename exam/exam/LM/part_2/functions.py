import torch
import torch.nn as nn

class VariationalDropout(nn.Module):
    def __init__(self, dropout):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training:
            return x
        else:
            mask = torch.empty_like(x).bernoulli_(1 - self.dropout)
            return x * mask / (1 - self.dropout)