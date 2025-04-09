import math

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
        #encoder_input = self.ln_head_encoder(encoder_input)
        encoder_input = self.bn_head_encoder(encoder_input)
        encoder_output = self.encoder(encoder_input)

        # Embedding
        decoder_in = self.positional_encoding(self.tgt_tok_emb(decoder_in))
        decoder_in = F.dropout(decoder_in, p=self.dropout)

        # Decoder
        decoder_in = self.ln_head_decoder(decoder_in)
        #decoder_in = self.bn_head_decoder(decoder_in)
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