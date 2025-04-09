import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from modeling.fusing import LayerNorm
from modeling.masking import PadMasking, FutureMasking
from modeling.attention import AttentionLayer, Past
from modeling.embedding import PositionalEmbedding, TokenEmbedding, PositionalEncoding, tokenEmbedding
from modeling.feedforward import PositionwiseFeedForward,MLP
from typing import Optional, Tuple, List, Union
import copy
from torch.nn import TransformerDecoderLayer,TransformerEncoderLayer,TransformerEncoder,TransformerDecoder


class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    decoder_in (*)        float           (..., decoder_in_len, dims)
    mask            bool            (..., seq_len, decoder_in_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., decoder_in_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)

    def forward(self,
                x: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        # Layer normalizations are performed before the layers respectively.

        a = self.ln_attn(x)
        a, memory = self.attn(a, a, a, memory, mask)

        x = x + a
        x = x + self.ff(self.ln_ff(x))

        return x


class GPT(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, heads, input_dim, padding_idx,
                 bidirectional=False, rate=4, name='Transformer_test'):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(padding_idx)
        self.future_masking = FutureMasking()

        self.LSTM = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=4, dropout=dropout,
                            bidirectional=bidirectional)
        self.bn_head = nn.BatchNorm1d(81)
        self.positional_embedding = PositionalEmbedding(max_word_len, hidden_dim)
        self.token_embedding = TokenEmbedding(num_classes, hidden_dim)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, hidden_dim, rate, dropout)
            for _ in range(layers)])
        self.ln_head = LayerNorm(hidden_dim)
        self.dropout = dropout
        self.Linear = nn.Linear(input_dim, hidden_dim)
        self.outlinear = nn.Linear(hidden_dim, num_classes)

    def forward(self,
                spectra: torch.Tensor,
                decoder_in: torch.Tensor = None):
        batch_size = spectra.size(0)

        x = decoder_in
        offset = spectra.size(-2) if spectra is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x)
        x = self.dropout_embedding(x)

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.LSTM.num_layers * 2) if self.bidirectional else self.LSTM.num_layers, batch_size,
                        self.LSTM.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (_, _) = self.LSTM(spectra, (hc, hc))
        #encoder_outputs = self.bn_head(encoder_outputs)

        # Apply transformer layers sequentially.
        for i, transformer in enumerate(self.transformers):
            x = transformer(x, encoder_outputs, mask)
        x = self.ln_head(x)
        #x = F.dropout(x, p=self.dropout)
        x = F.log_softmax(self.outlinear(x), dim=-1)

        return x

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, _ = batch
        batch_size = spectra.shape[0]

        inp = torch.zeros((batch_size, self.max_word_len)).type_as(inputs)
        outs = torch.empty((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            outputs = self(spectra, inp)
            outs[:, i] = outputs[:, i]
            if i < self.max_word_len-1:
                inp[:, (i + 1)] = outputs[:, i].argmax(dim=-1)
        return outs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer


class Transformer_mixture(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, heads, input_dim, padding_idx,
                 bidirectional=False, rate=4, name='Transformer_test'):
        super(Transformer_mixture, self).__init__()
        self.save_hyperparameters()
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(padding_idx)
        self.future_masking = FutureMasking()

        self.LSTM = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=4, dropout=dropout,
                            bidirectional=bidirectional)
        self.bn_head = nn.BatchNorm1d(81)
        self.positional_embedding = PositionalEmbedding(max_word_len, hidden_dim)
        self.token_embedding = TokenEmbedding(num_classes, hidden_dim)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers_1 = nn.ModuleList([
            TransformerLayer(heads, hidden_dim, rate, dropout)
            for _ in range(layers)])
        self.transformers_2 = nn.ModuleList([
            TransformerLayer(heads, hidden_dim, rate, dropout)
            for _ in range(layers)])
        self.ln_head = LayerNorm(hidden_dim)

        self.Linear = nn.Linear(input_dim, hidden_dim)
        self.outlinear = nn.Linear(hidden_dim, num_classes)

    def forward_1(self,
                spectra: torch.Tensor,
                decoder_in: torch.Tensor = None):
        batch_size = spectra.size(0)

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.LSTM.num_layers * 2) if self.bidirectional else self.LSTM.num_layers, batch_size,
                        self.LSTM.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (_, _) = self.LSTM(spectra, (hc, hc))

        x1 = decoder_in
        offset = spectra.size(-2) if spectra is not None else 0

        # Create masking tensor.
        mask1 = self.pad_masking(x1, offset)
        if not self.bidirectional:
            mask1 = mask1 + self.future_masking(x1, offset)

        # Use token embedding and positional embedding layers.
        x1 = self.token_embedding(x1) + self.positional_embedding(x1)
        x1 = self.dropout_embedding(x1)

        # Apply transformer layers sequentially.
        for i, transformer in enumerate(self.transformers_1):
            x1 = transformer(x1, encoder_outputs, mask1)
        x1 = self.ln_head(x1)

        x1 = F.log_softmax(self.outlinear(x1), dim=-1)

        return x1

    def forward_2(self,
                  spectra: torch.Tensor,
                  decoder_in: torch.Tensor = None):
        batch_size = spectra.size(0)

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.LSTM.num_layers * 2) if self.bidirectional else self.LSTM.num_layers, batch_size,
                        self.LSTM.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (_, _) = self.LSTM(spectra, (hc, hc))

        x2 = decoder_in
        offset = spectra.size(-2) if spectra is not None else 0

        # Create masking tensor.
        mask2 = self.pad_masking(x2, offset)
        if not self.bidirectional:
            mask2 = mask2 + self.future_masking(x2, offset)

        # Use token embedding and positional embedding layers.
        x2 = self.token_embedding(x2) + self.positional_embedding(x2)
        x2 = self.dropout_embedding(x2)

        for i, transformer in enumerate(self.transformers_2):
            x2 = transformer(x2, encoder_outputs, mask2)
        x2 = self.ln_head(x2)

        x2 = F.log_softmax(self.outlinear(x2), dim=-1)

        return x2

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, input1, input2, target1, target2 = batch
        output1 = self.forward_1(spectra, input1)
        output1 = output1.transpose(1, 2)
        output2 = self.forward_2(spectra, input2)
        output2 = output2.transpose(1, 2)

        loss1 = F.nll_loss(output1, target1, ignore_index=self.padding_idx)
        loss2 = F.nll_loss(output2, target2, ignore_index=self.padding_idx)

        loss = loss1 + loss2

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, input1, input2, target1, target2 = batch
        output1 = self.forward_1(spectra, input1)
        output1 = output1.transpose(1, 2)
        output2 = self.forward_2(spectra, input2)
        output2 = output2.transpose(1, 2)

        loss1 = F.nll_loss(output1, target1, ignore_index=self.padding_idx)
        loss2 = F.nll_loss(output2, target2, ignore_index=self.padding_idx)

        loss = loss1 + loss2

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return output1, output2

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, input1, input2, target1, target2 = batch
        output1 = self.forward_1(spectra, input1)
        output1 = output1.transpose(1, 2)
        output2 = self.forward_2(spectra, input2)
        output2 = output2.transpose(1, 2)
        return output1, output2

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, input1, input2, target1, target2 = batch
        batch_size = spectra.shape[0]

        inp1 = torch.zeros((batch_size, self.max_word_len)).type_as(input1)
        inp2 = torch.zeros((batch_size, self.max_word_len)).type_as(input1)
        out1 = torch.empty((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        out2 = torch.empty((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            output1 = self.forward_1(spectra, inp1)
            out1[:, i] = output1[:, i]
            if i < 23:
                inp1[:, (i + 1)] = output1[:, i].argmax(dim=-1)

        for i in range(self.max_word_len):
            output2 = self.forward_2(spectra, inp2)
            out2[:, i] = output2[:, i]
            if i < 23:
                inp2[:, (i + 1)] = output2[:, i].argmax(dim=-1)
        return out1, out2

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-6)
        return optimizer