import torch
from torch import nn
import numpy as np
import math


def t2v(tau, f, weight_linear, bias_linear, weight_periodic, bias_periodic, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, weight_linear) + bias_linear, arg)
    else:
        # print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, weight_linear) + bias_linear)

    v2 = torch.matmul(tau, weight_periodic) + bias_periodic
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, output_features):
        super(SineActivation, self).__init__()
        self.output_features = output_features
        self.weight_linear = nn.parameter.Parameter(torch.randn(in_features, output_features))
        self.bias_linear = nn.parameter.Parameter(torch.randn(output_features))
        self.weight_periodic = nn.parameter.Parameter(torch.randn(in_features, output_features))
        self.bias_periodic = nn.parameter.Parameter(torch.randn(output_features))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(
            tau,
            self.f,
            self.weight_linear,
            self.bias_linear,
            self.weight_periodic,
            self.bias_periodic,
        )


class CosineActivation(nn.Module):
    def __init__(self, in_features, output_features):
        super(CosineActivation, self).__init__()
        self.output_features = output_features
        self.weight_linear = nn.parameter.Parameter(torch.randn(in_features, output_features))
        self.bias_linear = nn.parameter.Parameter(torch.randn(output_features))
        self.weight_periodic = nn.parameter.Parameter(torch.randn(in_features, output_features))
        self.bias_periodic = nn.parameter.Parameter(torch.randn(output_features))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(
            tau,
            self.f,
            self.weight_linear,
            self.bias_linear,
            self.weight_periodic,
            self.bias_periodic,
        )


if __name__ == "__main__":
    sineact = SineActivation(1, 64)
    cosact = CosineActivation(1, 64)

    print(sineact(torch.Tensor([[7]])).shape)
    print(cosact(torch.Tensor([[7]])).shape)
