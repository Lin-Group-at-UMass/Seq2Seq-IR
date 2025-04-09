import os
import json
import torch
import numpy as np
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class SpectrumDataset(Dataset):
    '''

    PyTorch Dataset of IR Spectra and correspond selfies

    '''
    def __init__(self, path='./data/spectra_dataset.npz'):
        super(SpectrumDataset, self).__init__()

        # Dataset
        self.path = path
        self.data = np.load(self.path)

        # Symbol to integer label maps
        self.symbol_to_idx = json.load(open('./data/symbol_to_idx.json'))
        self.idx_to_symbol = {int(k):v for k, v in json.load(open('./data/idx_to_symbol.json')).items()}

        # IR spectra
        self.spectra = self.data['spectra']

        # Decoder inputs (SELFIES strings [START]...)
        self.decoder_inputs = self.data['decoderin']

        # Decoder outputs (SELFIES strins ...[STOP])
        self.decoder_outputs = self.data['decoderout']

        # Maximum SELFIES string length
        self.pad_to_len = max(sf.len_selfies(s) for s in self.decoder_inputs)

        # Calculate min and max for normalization
        self.spectra_min = np.min(self.spectra)
        self.spectra_max = np.max(self.spectra)

    def __len__(self):
        # Get number of samples
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        # Transform spectra to pytorch float tensors

        # Normalize the spectrum
        spectrum = self.spectra[idx]
        #spectrum = (spectrum - self.spectra_min) / (self.spectra_max - self.spectra_min)
        spectrum = torch.from_numpy(spectrum).float()

        # Transform decoder inputs to integer label vectors and to pytorch tensors
        decoder_input = self.decoder_inputs[idx]
        decoder_input = torch.as_tensor(sf.selfies_to_encoding(decoder_input, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        # Transform decoder outputs to integer label vectors and to pytorch tensors
        decoder_output = self.decoder_outputs[idx]
        decoder_output = torch.as_tensor(sf.selfies_to_encoding(decoder_output, self.symbol_to_idx, self.pad_to_len, enc_type='label'))

        return spectrum, decoder_input, decoder_output


class SpectrumDataModule(pl.LightningDataModule):
    '''

    SpectrumDataModule for SpectrumDataset

    '''
    def __init__(self, dataset, batch_size, test_split, val_split):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

        num_train = round(len(self.dataset) * (1 - self.val_split))
        num_val = len(self.dataset) - num_train
        self.train_data, val_set = random_split(self.dataset, [num_train, num_val])

        num_val = round(len(val_set) * (1 - self.test_split))
        num_test = len(val_set) - num_val
        self.val_data, self.test_data = random_split(val_set, [num_val, num_test])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
