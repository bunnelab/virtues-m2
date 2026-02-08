import torch.nn as nn


def build_activation(act_fcn):
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }.get(act_fcn)


def build_feedforward(d_model, dim_feedforward, activation="gelu", dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        build_activation(activation),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )


def linear_block(in_dim, out_dim, dropout_rate=0.05):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout_rate))
