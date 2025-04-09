import os
import json
import torch
import numpy as np
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from transformers import BartTokenizer
from rdkit import Chem

class SpectrumDataset(Dataset):
    '''

    PyTorch Dataset of IR Spectra and correspond selfies

    '''
    def __init__(self, tokenizer_path = './tokenizer-smiles-bart/', path='./data/dataset_smiles.npz'):
        super(SpectrumDataset, self).__init__()

        # Dataset
        self.path = path
        self.data = np.load(self.path)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

        # IR spectra
        self.spectra = self.data['ir']

        # SMILES Strings
        self.decoder_input = self.data['smi'].tolist()
        self.decoder_input = [self.get_canonical_smile(input) for input in self.decoder_input]
        output = self.tokenizer(
            self.decoder_input,
            padding=True,
            max_length=45,
            truncation=True,
            return_tensors="pt",
        )

        self.labels = output["input_ids"][:, 1:].cuda() if torch.cuda.is_available() else output["input_ids"][:, 1:]
        self.inputs = output["input_ids"][:, :-1].cuda() if torch.cuda.is_available() else output["input_ids"][:, :-1]

        self.pad_to_len = max([len(o) for o in output["input_ids"]]) - 1

    def __len__(self):
        # Get number of samples
        return self.spectra.shape[0]

    def get_canonical_smile(self,testsmi):
        try:
            #testsmi = Chem.MolToSmiles(Chem.MolFromSmiles(testsmi), doRandom=True)
            #return testsmi
            mol = Chem.MolFromSmiles(testsmi)
            return Chem.MolToSmiles(mol)
        except:
            print("Cannot convert {} to canonical smiles")
            return testsmi

    def __getitem__(self, idx):
        # Transform spectra to pytorch float tensors
        spectrum = self.spectra[idx]
        spectrum = torch.from_numpy(spectrum).float()

        # Transform decoder inputs to integer label vectors and to pytorch tensors
        inputs = self.inputs[idx]
        targets = self.labels[idx]
        return spectrum, inputs, targets


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
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
