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
from src.fusing import LayerNorm
from src.masking import PadMasking, FutureMasking
from src.attention import AttentionLayer, Past
from src.fusing import LayerNorm as LN
from src.embedding import PositionalEmbedding, TokenEmbedding, PositionalEncoding, tokenEmbedding
from src.feedforward import PositionwiseFeedForward,MLP
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


class LSTM(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM'):
        super(LSTM, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in, hc):
        # Get batch size
        batch_size = spectra.shape[0]

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        embedded = self.embedding(decoder_in)
        embedded = F.dropout(embedded, p=self.dropout)

        # Decode spectrum
        output, (_, _) = self.decoder(embedded, (hidden, cell))

        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)

        return output

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        # Calculate loss
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
            if i < 23:
                inp[:, (i + 1)] = outputs[:, i].argmax(dim=-1)
        return outs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters())
        return optimizer


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    https://github.com/sooftware
    """

    def __init__(self, hidden_dim, bidirectional):
        super(DotProductAttention, self).__init__()
        self.attn_combine = nn.Linear((hidden_dim * 4) if bidirectional else (hidden_dim * 2), hidden_dim)

    def forward(self, query, value):
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)
        embedded = torch.cat((query, context), 2)
        embedded = F.gelu(self.attn_combine(embedded))

        return embedded


class AttentionLSTM(LSTM):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='Attention LSTM'):
        super(LSTM, self).__init__()
        # Module properties
        self.save_hyperparameters()
        self.name = name

        # Maximum SELFIES string length
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

        # Encoder functions
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)

        # Decoder functions
        self.embedding = nn.Embedding(self.num_classes, (hidden_dim * 2) if bidirectional else hidden_dim,
                                      padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.attention = DotProductAttention(hidden_dim, bidirectional)
        self.batch_size = batch_size
        self.dropout = dropout

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        print(spectra.size())
        print(decoder_in.size())
        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty(self.encoder.num_layers if not self.bidirectional else (self.encoder.num_layers * 2),
                        batch_size, self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))
        print(hidden.size())
        print(cell.size())
        # Embed and dropout decoder input
        embedded = self.embedding(decoder_in)
        embedded = F.dropout(embedded, p=self.dropout)
        embedded = self.attention(embedded, encoder_outputs)
        print(embedded.size())

        output, (_, _) = self.decoder(embedded, (hidden, cell))
        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)
        print(output.size())
        return output


class MLSTM(LSTM):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='MLSTM'):
        super(LSTM, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.feature_extract1 = nn.LSTM(2, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                                        bidirectional=bidirectional)
        self.feature_extract2 = nn.LSTM(81, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                                        bidirectional=bidirectional)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.attention = DotProductAttention(hidden_dim, bidirectional)
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)
        hc1 = nn.init.xavier_uniform_(
            torch.empty(
                (self.feature_extract1.num_layers * 2) if self.bidirectional else self.feature_extract1.num_layers,
                batch_size,
                self.feature_extract1.hidden_size)).type_as(spectra)
        hc2 = nn.init.xavier_uniform_(
            torch.empty(
                (self.feature_extract2.num_layers * 2) if self.bidirectional else self.feature_extract2.num_layers,
                batch_size,
                self.feature_extract2.hidden_size)).type_as(spectra)

        # Feature Extract
        feature1, (_, _) = self.feature_extract1(spectra, (hc1, hc1))
        feature2, (_, _) = self.feature_extract2(spectra.transpose(1, -1), (hc2, hc2))
        features = torch.cat((feature1, feature2), dim=1)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(features, (hc, hc))

        # Embed and dropout decoder input
        embedded = self.embedding(decoder_in)
        embedded = F.dropout(embedded, p=self.dropout)
        embedded = self.attention(embedded, encoder_outputs)

        # Decode spectrum
        output, (_, _) = self.decoder(embedded, (hidden, cell))

        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)

        return output


class GRU(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='GRU'):
        super(GRU, self).__init__()

        # Module properties
        self.save_hyperparameters()
        self.name = name

        # Maximum SELFIES string length
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

        # Encoder functions
        self.encoder = nn.GRU(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                              bidirectional=bidirectional)

        # Decoder functions
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        # self.attention = DotProductAttention(hidden_dim, bidirectional)
        self.batch_size = batch_size

        # Global functions
        self.dropout = dropout

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, hidden = self.encoder(spectra, hc)
        print(hidden.size())
        # Embed and dropout decoder input
        embedded = self.embedding(decoder_in)
        embedded = F.dropout(embedded, p=self.dropout)
        print(embedded.size())
        # Decode spectrum
        output, _ = self.decoder(embedded, hidden)
        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)
        return output

    def training_step(self, batch, batch_idx):
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, _ = batch
        batch_size = spectra.shape[0]

        inp = torch.ones((batch_size, self.max_word_len)).type_as(inputs)
        outs = torch.ones((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            outputs = self(spectra, inp)
            outs[:, i] = outputs[:, i]
            if i < 23:
                inp[:, (i + 1)] = outputs[:, i].argmax(dim=-1)
        return outs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters())
        return optimizer


class AttentionGRU(GRU):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx, bidirectional,
                 batch_size, name='Attention GRU'):
        super(GRU, self).__init__()

        # Module properties
        self.save_hyperparameters()
        self.name = name

        # Maximum SELFIES string length
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

        # Encoder functions
        self.encoder = nn.GRU(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                              bidirectional=bidirectional)

        # Decoder functions
        self.embedding = nn.Embedding(self.num_classes, (hidden_dim * 2) if bidirectional else hidden_dim,
                                      padding_idx=self.padding_idx)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.attention = DotProductAttention(hidden_dim, bidirectional)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_size = batch_size

        # Global functions
        self.dropout = dropout

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, hidden = self.encoder(spectra, hc)

        # Embed and dropout decoder input
        embedded = self.embedding(decoder_in)
        embedded = F.dropout(embedded, p=self.dropout)
        embedded = self.attention(embedded, encoder_outputs)

        # Decode spectrum
        output, _ = self.decoder(embedded, hidden)
        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, self.num_classes)

    def forward(self, embedded, out, hidden, cell, index):
        output, (_, _) = self.decoder(embedded, (hidden, cell))

        output = output[:, index:index + 1, :]
        output = F.dropout(output, p=self.dropout)
        output = F.gelu(self.linear1(output))
        output = F.log_softmax(self.linear2(output), dim=-1)

        out = torch.cat([out[:, :index, :], output, out[:, index + 1:, :]], dim=1)

        output = output.argmax(dim=-1)
        output = F.dropout(self.embedding(output), p=self.dropout)

        embedded = torch.cat([embedded[:, :index + 1, :], output, embedded[:, index + 2:, :]], dim=1)
        return embedded, out


class SpectraNet(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(SpectraNet, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)

        self.decoder = Decoder(hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx)

        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, self.max_word_len + 1)).type_as(decoder_in)
        embedded = self.embedding(inp)
        embedded = F.dropout(embedded, p=self.dropout)

        return embedded, hidden, cell

    def training_step(self, batch, batch_idx, optimizer_idx):
        beta = 0.2
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        loss = torch.zeros(1).type_as(spectra)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        length = 0

        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)
            output = out.transpose(1, 2)
            temp_loss = 0
            ignore = torch.full(targets[:, i:i + 1].size(), 31).type_as(targets)
            if torch.all(targets[:, i:i + 1] != ignore):
                temp_loss = F.nll_loss(output[:, :, i:i + 1], targets[:, i:i + 1], ignore_index=self.padding_idx)
                length += 1
            loss += temp_loss
        loss = loss / length

        outputs = out.transpose(1, 2)
        loss1 = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)

        if optimizer_idx == 0:
            loss1 = loss1 * (1 - beta) + loss * beta
            self.log('loss1', loss1, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": loss1}

        elif optimizer_idx == 1:
            loss = loss * (1 - beta) + loss1 * beta
            self.log('loss2', loss, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)

        outputs = out.transpose(1, 2)
        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)

        outputs = out.transpose(1, 2)
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)

        return out

    def configure_optimizers(self):
        # Configure optimizer
        optimizer1 = optim.Adam([{"params": self.encoder.parameters()},
                                 {"params": self.decoder.decoder.parameters()}], weight_decay=1e-6)
        optimizer2 = optim.Adam([{"params": self.decoder.linear1.parameters()},
                                 {"params": self.decoder.linear2.parameters()}], weight_decay=1e-6)
        return [optimizer1, optimizer2]


class SpectraNet_test(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(SpectraNet_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)

        self.decoder = Decoder(hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx)

        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, self.max_word_len + 1)).type_as(decoder_in)
        embedded = self.embedding(inp)
        embedded = F.dropout(embedded, p=self.dropout)

        return embedded, hidden, cell

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        loss = torch.zeros(1).type_as(spectra)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)

        length = 0

        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)
            output = out.transpose(1, 2)
            temp_loss = 0
            ignore = torch.full(targets[:, i:i + 1].size(), 31).type_as(targets)
            if torch.all(targets[:, i:i + 1] != ignore):
                temp_loss = F.nll_loss(output[:, :, i:i + 1], targets[:, i:i + 1])
                length += 1
            loss += temp_loss
        loss = loss / length
        outputs = out.transpose(1, 2)
        loss1 = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)

        if optimizer_idx == 0:
            self.log('loss1', loss1, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": loss1}

        elif optimizer_idx == 1:
            lossx = loss * 0.4 + loss1 * 0.6
            self.log('loss2', lossx, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": lossx}

        elif optimizer_idx == 2:
            loss = loss * 0.6 + loss1 * 0.4
            self.log('loss3', loss, on_step=True, on_epoch=True, prog_bar=True)
            return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)

        outputs = out.transpose(1, 2)
        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)
        outputs = out.transpose(1, 2)
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        batch_size = spectra.size(0)
        embedded, hidden, cell = self(spectra, inputs)
        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(spectra)
        for i in range(self.max_word_len):
            embedded, out = self.decoder(embedded, out, hidden, cell, i)

        return out

    def configure_optimizers(self):
        # Configure optimizer
        optimizer1 = optim.Adam(self.encoder.parameters(), weight_decay=1e-6)
        optimizer2 = optim.Adam(self.decoder.decoder.parameters(), weight_decay=1e-6)
        optimizer3 = optim.Adam(self.decoder.linear.parameters(), weight_decay=1e-6)
        return [optimizer1, optimizer2, optimizer3]


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hidden_dim, temperature=8, attn_dropout=0.1):
        super().__init__()
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        input = v
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        attn = self.dropout(F.softmax(attn, dim=-1))
        context = torch.matmul(attn, v)
        context = self.dropout(F.gelu(self.fc(context)))
        output = context + input

        return output


class LSTM_test(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(LSTM_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)

        # self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear2 = nn.Linear(hidden_dim, self.num_classes)
        self.train_loss = 0
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, self.max_word_len)).type_as(spectra)
        inp = inp.int()
        embedded = self.embedding(inp)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(embedded)

        # Decode spectrum
        for i in range(self.max_word_len):
            output, hidden, cell = self.forward_step(embedded, hidden, cell)
            output = output[:, i:i + 1, :]
            output = self.linear2(output)

            if decoder_in is not None:
                inp = decoder_in[:, i + 1:i + 2]
                inp = self.embedding(inp)
            else:
                _, top1 = output.topk(1)
                inp = top1.squeeze(-1).detach()
                inp = self.embedding(inp)

            embedded = torch.cat([embedded[:, :i, :], inp, embedded[:, i + 1:, :]], dim=1)

            output = F.log_softmax(output, dim=-1)
            out = torch.cat([out[:, :i, :], output, out[:, i + 1:, :]], dim=1)

        return out

    def forward_step(self, inp, hidden, cell, model=None):
        output = F.gelu(inp)
        if model is not None:
            output, (hidden, cell) = model(output, (hidden, cell))
        else:
            output, (hidden, cell) = self.decoder(output, (hidden, cell))
        output = F.dropout(output, p=self.dropout)
        return output, hidden, cell

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
        # print("Train Loss: ", self.train_loss)
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
        outputs = self(spectra, None)
        return outputs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-6)
        return optimizer


class LSTM_autoregressive(LSTM_test):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(LSTM_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)

        # self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear2 = nn.Linear(hidden_dim, self.num_classes)
        self.train_loss = 0
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        # spectra = self.ScaledDotProductAttention1(spectra, spectra, spectra)
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, 1)).type_as(spectra)
        inp = inp.int()
        embedded = self.embedding(inp)
        embedded = F.dropout(embedded, p=self.dropout)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(embedded)

        # Decode spectrum
        for i in range(self.max_word_len):
            output, hidden, cell = self.forward_step(embedded, hidden, cell)
            output = F.dropout(output, p=self.dropout)
            output = self.linear2(output)

            if decoder_in is not None:
                inp = decoder_in[:, i + 1:i + 2]
            else:
                _, top1 = output.topk(1)
                inp = top1.squeeze(-1).detach()

            embedded = self.embedding(inp)
            embedded = F.dropout(embedded, p=self.dropout)

            output = F.log_softmax(output, dim=-1)
            out = torch.cat([out[:, :i, :], output, out[:, i + 1:, :]], dim=1)
        return out

class test_1(LSTM_test):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(LSTM_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)

        # self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear2 = nn.Linear(hidden_dim, self.num_classes)
        self.train_loss = 0
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, 1)).type_as(spectra)
        inp = inp.int()
        embedded = self.embedding(inp)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(embedded)

        # Decode spectrum
        for i in range(self.max_word_len):
            output, hidden, cell = self.forward_step(embedded, hidden, cell)
            output = output[:, -1:, :]
            output = self.linear2(output)

            if decoder_in is not None:
                inp = decoder_in[:, i + 1:i + 2]
                inp = self.embedding(inp)
            else:
                _, top1 = output.topk(1)
                inp = top1.squeeze(-1).detach()
                inp = self.embedding(inp)

            embedded = inp

            output = F.log_softmax(output, dim=-1)
            out = torch.cat([out[:, :i, :], output, out[:, i + 1:, :]], dim=1)

        return out
class LSTM_autoregressive_attention(LSTM_test):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(LSTM_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.ScaledDotProductAttention1 = ScaledDotProductAttention(self.input_dim)
        self.encoder = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.ScaledDotProductAttention2 = ScaledDotProductAttention(self.hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                               num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)
        self.ScaledDotProductAttention3 = ScaledDotProductAttention(self.hidden_dim)

        # self.linear1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear2 = nn.Linear(hidden_dim, self.num_classes)
        self.train_loss = 0
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        #spectra = self.ScaledDotProductAttention1(spectra, spectra, spectra)
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, 1)).type_as(spectra)
        inp = inp.int()
        embedded = self.embedding(inp)
        embedded = F.dropout(embedded, p=self.dropout)
        embedded = self.ScaledDotProductAttention2(embedded, embedded, embedded)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(embedded)

        # Decode spectrum
        for i in range(self.max_word_len):
            output, hidden, cell = self.forward_step(embedded, hidden, cell)
            output = F.dropout(output, p=self.dropout)
            output = self.ScaledDotProductAttention3(output, output, output)
            output = self.linear2(output)

            if decoder_in is not None:
                inp = decoder_in[:, i + 1:i + 2]
            else:
                _, top1 = output.topk(1)
                inp = top1.squeeze(-1).detach()

            embedded = self.embedding(inp)
            embedded = F.dropout(embedded, p=self.dropout)
            embedded = self.ScaledDotProductAttention2(embedded, embedded, embedded)

            output = F.log_softmax(output, dim=-1)
            out = torch.cat([out[:, :i, :], output, out[:, i + 1:, :]], dim=1)

        return out

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
        #self.bn_head = nn.BatchNorm1d(81)
        self.positional_embedding = PositionalEmbedding(max_word_len, hidden_dim)
        self.token_embedding = TokenEmbedding(num_classes, hidden_dim)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(heads, hidden_dim, rate, dropout)
            for _ in range(layers)])
        self.ln_head = LN(hidden_dim)
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

class Transformer(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,heads,
                 bidirectional=False, batch_size=64, name='LSTM'):
        super(Transformer, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

        self.mlp = MLP(input_dim, 4, hidden_dim, hidden_dim, activation=nn.SiLU())

        self.encoderlayer = TransformerEncoderLayer(d_model=hidden_dim,nhead=heads,dim_feedforward=hidden_dim*4,
                                                    dropout=dropout,batch_first=True, norm_first=False)
        self.encoder = TransformerEncoder(encoder_layer=self.encoderlayer, num_layers=layers)
        self.ln_head_encoder = nn.LayerNorm(hidden_dim)
        self.bn_head_encoder = nn.BatchNorm1d(81)

        self.tgt_tok_emb = tokenEmbedding(self.num_classes, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0)

        self.decoderlayer = TransformerDecoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim*4,
                                                    dropout=dropout, batch_first=True, norm_first=False)
        self.decoder = TransformerDecoder(decoder_layer=self.decoderlayer, num_layers=layers)
        self.ln_head_decoder = nn.LayerNorm(hidden_dim)
        self.bn_head_decoder = nn.BatchNorm1d(self.max_word_len)

        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.dropout = dropout
        self.batch_size = batch_size

    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device="cuda" if torch.cuda.is_available() else 'cpu')) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self,tgt):
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = (tgt == self.padding_idx)
        return tgt_mask, tgt_padding_mask

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]

        tgt_mask, tgt_key_padding_mask = self.create_mask(decoder_in)

        # Encode spectrum
        encoder_input = self.mlp(spectra)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_input = self.bn_head_encoder(encoder_input)
        encoder_output = self.encoder(encoder_input)

        # Embedding
        decoder_in = self.positional_encoding(self.tgt_tok_emb(decoder_in))
        decoder_in = F.dropout(decoder_in, p=self.dropout)

        # Decoder
        decoder_in = self.ln_head_decoder(decoder_in)
        output = self.decoder(tgt=decoder_in, memory=encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)

        return output

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs)
        outputs = outputs.transpose(1, 2)

        # Calculate loss
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

class GRU_test(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM_test'):
        super(GRU_test, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                              bidirectional=bidirectional)
        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)

        self.linear2 = nn.Linear(hidden_dim, self.num_classes)
        self.train_loss = 0
        self.dropout = dropout
        self.batch_size = batch_size

    def forward(self, spectra, decoder_in):
        # Get batch size
        batch_size = spectra.shape[0]
        structure = []

        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, batch_size,
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        encoder_outputs, hidden = self.encoder(spectra, hc)

        # Embed and dropout decoder input
        inp = torch.zeros((batch_size, self.max_word_len)).type_as(spectra)
        inp = inp.int()
        embedded = self.embedding(inp)

        out = torch.zeros((batch_size, self.max_word_len, self.num_classes)).type_as(embedded)

        # Decode spectrum
        for i in range(self.max_word_len):
            output, hidden = self.forward_step(embedded, hidden)
            output = output[:, i:i + 1, :]
            output = self.linear2(output)

            if decoder_in is not None:
                inp = decoder_in[:, i + 1:i + 2]
                inp = self.embedding(inp)
            else:
                _, top1 = output.topk(1)
                inp = top1.squeeze(-1).detach()
                inp = self.embedding(inp)

            embedded = torch.cat([embedded[:, :i, :], inp, embedded[:, i + 1:, :]], dim=1)

            output = F.log_softmax(output, dim=-1)
            out = torch.cat([out[:, :i, :], output, out[:, i + 1:, :]], dim=1)

        return out

    def forward_step(self, inp, hidden, model=None):
        output = F.gelu(inp)
        if model is not None:
            output, hidden = model(output, hidden)
        else:
            output, hidden = self.decoder(output, hidden)
        output = F.dropout(output, p=self.dropout)
        return output, hidden

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
        # print("Train Loss: ", self.train_loss)
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
        outputs = self(spectra, None)
        return outputs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters(), weight_decay=1e-6)
        return optimizer