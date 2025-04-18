import os
import json
import torch
import numpy as np
import selfies as sf
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer
from rdkit import Chem


class SelfiesSpectrumDataset(Dataset):
    """
    PyTorch Dataset of IR spectra and corresponding SELFIES.
    """
    def __init__(self, path='./data/spectra_dataset.npz',
                 symbol_to_idx_path='./data/symbol_to_idx.json',
                 idx_to_symbol_path='./data/idx_to_symbol.json'):
        super().__init__()
        self.data = np.load(path)
        self.symbol_to_idx = json.load(open(symbol_to_idx_path))
        self.idx_to_symbol = {int(k): v for k, v in json.load(open(idx_to_symbol_path)).items()}
        self.spectra = self.data['spectra']
        self.decoder_inputs = self.data['decoderin']
        self.decoder_outputs = self.data['decoderout']
        self.pad_to_len = max(sf.len_selfies(s) for s in self.decoder_inputs)
        # normalization parameters
        self.spectra_min = np.min(self.spectra)
        self.spectra_max = np.max(self.spectra)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        # optional normalization:
        # spectrum = (spectrum - self.spectra_min) / (self.spectra_max - self.spectra_min)
        spectrum = torch.from_numpy(spectrum).float()

        decoder_input = self.decoder_inputs[idx]
        decoder_input = torch.tensor(
            sf.selfies_to_encoding(decoder_input, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )
        decoder_output = self.decoder_outputs[idx]
        decoder_output = torch.tensor(
            sf.selfies_to_encoding(decoder_output, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )
        return spectrum, decoder_input, decoder_output


class SelfiesSpectrumDataModule(pl.LightningDataModule):
    """
    DataModule for SelfiesSpectrumDataset.
    """
    def __init__(self, dataset, batch_size=32, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        num_train = int(len(self.dataset) * (1 - self.val_split))
        num_val = len(self.dataset) - num_train
        train_data, val_test = random_split(self.dataset, [num_train, num_val])

        num_val2 = int(len(val_test) * (1 - self.test_split))
        num_test = len(val_test) - num_val2
        self.val_data, self.test_data = random_split(val_test, [num_val2, num_test])
        self.train_data = train_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self):
        return self.val_dataloader()


class MixtureSpectrumDataset(Dataset):
    """
    PyTorch Dataset of IR spectra and corresponding mixture SELFIES.
    Returns: spectrum, decoder_input1, decoder_input2, decoder_output1, decoder_output2.
    """
    def __init__(self, path='./data/randomized-combined_dataset.npz',
                 symbol_to_idx_path='./data/symbol_to_idx_mixture.json',
                 idx_to_symbol_path='./data/idx_to_symbol_mixture.json'):
        super().__init__()
        self.data = np.load(path)
        self.symbol_to_idx = json.load(open(symbol_to_idx_path))
        self.idx_to_symbol = {int(k): v for k, v in json.load(open(idx_to_symbol_path)).items()}
        self.spectra = self.data['spectra']
        self.decoder_inputs = self.data['decoderin']
        self.decoder_outputs = self.data['decoderout']
        self.pad_to_len = max(sf.len_selfies(s) for s in self.decoder_inputs)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = torch.from_numpy(self.spectra[idx]).float()

        inputs = self.decoder_inputs[idx].split('[SPACE]')
        inp1, inp2_raw = inputs[0], inputs[1]
        inp2 = '[START]' + inp2_raw

        decoder_input1 = torch.tensor(
            sf.selfies_to_encoding(inp1, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )
        decoder_input2 = torch.tensor(
            sf.selfies_to_encoding(inp2, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )

        outputs = self.decoder_outputs[idx].split('[SPACE]')
        out1_raw, out2 = outputs[0], outputs[1]
        out1 = out1_raw + '[STOP]'

        decoder_output1 = torch.tensor(
            sf.selfies_to_encoding(out1, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )
        decoder_output2 = torch.tensor(
            sf.selfies_to_encoding(out2, self.symbol_to_idx, self.pad_to_len, enc_type='label'),
            dtype=torch.long
        )

        return spectrum, decoder_input1, decoder_input2, decoder_output1, decoder_output2


class MixtureSpectrumDataModule(pl.LightningDataModule):
    """
    DataModule for MixtureSpectrumDataset.
    """
    def __init__(self, dataset, batch_size=32, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        num_train = int(len(self.dataset) * (1 - self.val_split))
        num_val = len(self.dataset) - num_train
        train_data, val_test = random_split(self.dataset, [num_train, num_val])

        num_val2 = int(len(val_test) * (1 - self.test_split))
        num_test = len(val_test) - num_val2
        self.val_data, self.test_data = random_split(val_test, [num_val2, num_test])
        self.train_data = train_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self):
        return self.val_dataloader()


class SmilesSpectrumDataset(Dataset):
    """
    PyTorch Dataset of IR spectra and corresponding SMILES with BART tokenizer.
    """
    def __init__(self, path='./data/dataset_smiles.npz', tokenizer_path='./tokenizer-smiles/'):
        super().__init__()
        self.data = np.load(path)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.spectra = self.data['ir']
        smi_list = self.data['smi'].tolist()
        self.decoder_inputs = [self._canonical_smile(s) for s in smi_list]
        tokenized = self.tokenizer(self.decoder_inputs, padding=True, max_length=45,
                                   truncation=True, return_tensors='pt')
        input_ids = tokenized['input_ids']
        self.inputs = input_ids[:, :-1]
        self.labels = input_ids[:, 1:]
        self.pad_to_len = input_ids.size(1) - 1

    def __len__(self):
        return len(self.spectra)

    def _canonical_smile(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol)
        except Exception:
            print(f"Cannot convert {smi} to canonical SMILES")
            return smi

    def __getitem__(self, idx):
        spectrum = torch.from_numpy(self.spectra[idx]).float()
        return spectrum, self.inputs[idx], self.labels[idx]


class SmilesSpectrumDataModule(pl.LightningDataModule):
    """
    DataModule for SmilesSpectrumDataset.
    """
    def __init__(self, dataset, batch_size=32, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        num_train = int(len(self.dataset) * (1 - self.val_split))
        num_val = len(self.dataset) - num_train
        train_data, val_test = random_split(self.dataset, [num_train, num_val])

        num_val2 = int(len(val_test) * (1 - self.test_split))
        num_test = len(val_test) - num_val2
        self.val_data, self.test_data = random_split(val_test, [num_val2, num_test])
        self.train_data = train_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count(), pin_memory=True)

    def predict_dataloader(self):
        return self.val_dataloader()
