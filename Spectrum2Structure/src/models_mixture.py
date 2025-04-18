import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from src.fusing import LayerNorm as LN
from src.models import TransformerLayer
from src.masking import PadMasking, FutureMasking
from src.attention import AttentionLayer, Past
from src.embedding import PositionalEmbedding, TokenEmbedding, PositionalEncoding, tokenEmbedding
from src.feedforward import PositionwiseFeedForward,MLP
from typing import Optional, Tuple, List, Union
import copy
from torch.nn import TransformerDecoderLayer,TransformerEncoderLayer,TransformerEncoder,TransformerDecoder


class LSTM_Mixture(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM'):
        super(LSTM_Mixture, self).__init__()
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

    def Encoder(self, spectra):
        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hc = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, spectra.size(0),
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        # spectra = [batch size, 81, 2]
        encoder_outputs, (hidden, cell) = self.encoder(spectra, (hc, hc))

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

    def Decoder(self, decoder_input, hidden, cell):
        # Embed and dropout decoder input
        embedded = self.embedding(decoder_input)
        embedded = F.dropout(embedded, p=self.dropout)

        # Decode spectrum
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))

        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)

        return output, hidden, cell


    def forward(self, spectra, decoder_in, mode="Training"):
        if mode=='Training' or mode=='Validation' or mode=='Testing':
            hidden, cell = self.Encoder(spectra)
            output, hidden, cell = self.Decoder(decoder_in, hidden, cell)
            return output
        elif mode=="Prediction":
            inp = torch.zeros((decoder_in.size(0), 1)).type_as(decoder_in)
            outputs = torch.empty((spectra.size(0), self.max_word_len, self.num_classes)).type_as(spectra)
            hidden, cell = self.Encoder(spectra)
            for i in range(self.max_word_len):
                output, hidden, cell = self.Decoder(inp, hidden, cell)
                outputs[:, i] = output.squeeze(1)
                if i<self.max_word_len-1:
                    _, top1 = output.topk(1)
                    inp = top1.squeeze(-1).detach()
            return outputs

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Training')
        outputs = outputs.transpose(1, 2)

        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Validation')
        outputs = outputs.transpose(1, 2)

        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Testing')
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, _ = batch

        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode="Prediction")
        return outputs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters())
        return optimizer


class GRU_Mixture(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,
                 bidirectional=False, batch_size=64, name='LSTM'):
        super(GRU_Mixture, self).__init__()
        self.save_hyperparameters()
        self.name = name
        self.max_word_len = max_word_len
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(self.input_dim, hidden_dim, batch_first=True, num_layers=layers, dropout=dropout,
                               bidirectional=bidirectional)

        self.embedding = nn.Embedding(self.num_classes, hidden_dim, padding_idx=self.padding_idx)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=(layers * 2) if bidirectional else layers, dropout=dropout)

        self.linear = nn.Linear(hidden_dim, self.num_classes)
        self.dropout = dropout
        self.batch_size = batch_size

    def Encoder(self, spectra):
        # Initialize hidden and cell state tensors (xavier uniform distribution)
        hidden = nn.init.xavier_uniform_(
            torch.empty((self.encoder.num_layers * 2) if self.bidirectional else self.encoder.num_layers, spectra.size(0),
                        self.encoder.hidden_size)).type_as(spectra)

        # Encode spectrum
        # spectra = [batch size, 81, 2]
        encoder_outputs, hidden = self.encoder(spectra, hidden)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden

    def Decoder(self, decoder_input, hidden):
        # Embed and dropout decoder input
        embedded = self.embedding(decoder_input)
        embedded = F.dropout(embedded, p=self.dropout)

        # Decode spectrum
        output, hidden = self.decoder(embedded, hidden)

        output = F.dropout(output, p=self.dropout)

        # Project to vocabulary size
        output = F.log_softmax(self.linear(output), dim=-1)

        return output, hidden


    def forward(self, spectra, decoder_in, mode="Training"):
        if mode=='Training' or mode=='Validation' or mode=='Testing':
            hidden = self.Encoder(spectra)
            output, hidden = self.Decoder(decoder_in, hidden)
            return output
        elif mode=="Prediction":
            inp = torch.zeros((decoder_in.size(0), 1)).type_as(decoder_in)
            outputs = torch.empty((spectra.size(0), self.max_word_len, self.num_classes)).type_as(spectra)
            hidden = self.Encoder(spectra)
            for i in range(self.max_word_len):
                output, hidden = self.Decoder(inp, hidden)
                outputs[:, i] = output.squeeze(1)
                if i<self.max_word_len-1:
                    _, top1 = output.topk(1)
                    inp = top1.squeeze(-1).detach()
            return outputs

    def training_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Training')
        outputs = outputs.transpose(1, 2)

        # Calculate loss
        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Validation')
        outputs = outputs.transpose(1, 2)

        loss = F.nll_loss(outputs, targets, ignore_index=self.padding_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode='Testing')
        return outputs

    def predict_step(self, batch, batch_idx):
        # Get predictions
        spectra, inputs, _ = batch

        spectra, inputs, targets = batch
        outputs = self(spectra, inputs, mode="Prediction")
        return outputs

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = optim.Adam(self.parameters())
        return optimizer

class GPT_Mixture(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, heads, input_dim, padding_idx,
                 bidirectional=False, rate=4, name='Transformer_test'):
        super(GPT_Mixture, self).__init__()
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

class Transformer_Mixture(pl.LightningModule):
    def __init__(self, hidden_dim, dropout, layers, max_word_len, num_classes, input_dim, padding_idx,heads,
                 bidirectional=False, batch_size=64, name='LSTM'):
        super(Transformer_Mixture, self).__init__()
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
        self.bn_head_encoder = nn.BatchNorm1d(162)

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