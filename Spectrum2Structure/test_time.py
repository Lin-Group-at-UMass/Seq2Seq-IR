import sys
import warnings
from models import LSTM,GRU,AttentionLSTM,AttentionGRU,LSTM_autoregressive,LSTM_autoregressive_attention,GPT,Transformer
import pytorch_lightning as pl
import numpy as np
from dataset_utils import SpectrumDataset, SpectrumDataModule
#from dataset_utils_mixture import SpectrumDataset_2, SpectrumDataModule
from tqdm import tqdm
import torch.nn.functional as F
import torch
import selfies as sf
from rdkit import Chem
from rdkit import DataStructs
from torchmetrics.functional import confusion_matrix, jaccard_index, cosine_similarity, hamming_distance
from rdkit.Chem import AllChem
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_InchiKey(smi):
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key
    except:
        return None


def judge_InchiKey(key1, key2):
    if key1 is None or key2 is None:
        return False
    return key1 == key2


def same_smi(smi1, smi2):
    key1 = get_InchiKey(smi1)
    if key1 is None:
        return False
    key2 = get_InchiKey(smi2)
    if key2 is None:
        return False
    return judge_InchiKey(key1, key2)

#torch.cuda.set_device(0)

# Set seed and deterministic behavior
seed = 78438379
deterministic = True
seed_everything(seed, workers=deterministic)
torch.use_deterministic_algorithms(deterministic)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set dataloader parameters
batch_size = 256
val_split = 0.2
test_split = 0.5

# Set model parameters (found in hparams.yaml file)
hidden_dim = 768
dropout = 0.1
layers = 6
heads = 6

# Set up dataset and Datamodule
dataset = SpectrumDataset()
dataloader = SpectrumDataModule(dataset=dataset, batch_size=batch_size, test_split=test_split, val_split=val_split)

max_word_len = dataset.pad_to_len
num_classes = len(dataset.symbol_to_idx)
input_dim = dataset.spectra.shape[-1]
padding_idx = dataset.symbol_to_idx['[nop]']

# define the model
model=Transformer.load_from_checkpoint("checkpoints/Transformer-epoch=96-step=30070.ckpt", map_location="cpu")

import time

start_time = time.time()

trainer = pl.Trainer()
predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))

end_time = time.time()

execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")