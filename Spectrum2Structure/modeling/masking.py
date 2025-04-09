import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])


class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)

        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])

class MemoryPadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (batch_size, seq_len)
    memory          long            (batch_size, mem_len)
    ---------------------------------------------------------------------------
    output          float           (batch_size, seq_len, mem_len)
    ===========================================================================
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        is_pad = (memory == self.pad_idx).unsqueeze(1)
        mask = is_pad.expand(x.size(0), x.size(1), memory.size(1))
        return mask


class MemoryFutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (batch_size, seq_len)
    memory          long            (batch_size, mem_len)
    ---------------------------------------------------------------------------
    output          float           (batch_size, seq_len, mem_len)
    ===========================================================================
    """
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mem_len = memory.size(1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, mem_len),
                            dtype=torch.bool, device=x.device)
        future = future.triu(1)

        mask = future.view((1,) + future.size())
        mask = mask.expand(x.size(0), seq_len, mem_len)
        return mask
