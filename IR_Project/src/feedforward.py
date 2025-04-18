import torch
import torch.nn as nn
import math


class Swish(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class PositionwiseFeedForward(nn.Sequential):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__(
            nn.Linear(dims, dims * rate),
            nn.GELU(),
            #Swish(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims))

class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)


class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        n_layers,
        d_hidden,
        d_out,
        activation=nn.ReLU(),
        bias=True,
        final_init="final",
    ):
        super(MLP, self).__init__()
        layers = [Linear(d_in, d_hidden, bias), activation]
        for _ in range(n_layers):
            layers += [Linear(d_hidden, d_hidden, bias), activation]
        layers.append(Linear(d_hidden, d_out, bias, init=final_init))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
