import sys

import warnings
from models import LSTM,GRU,AttentionLSTM,AttentionGRU,LSTM_autoregressive,LSTM_autoregressive_attention,GPT,Transformer
import pytorch_lightning as pl
from rdkit.Chem import Descriptors, rdMolDescriptors
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


#torch.cuda.set_device(0)

# Set seed and deterministic behavior
seed = 78438379
deterministic = True
seed_everything(seed, workers=deterministic)
torch.use_deterministic_algorithms(deterministic)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set dataloader parameters
batch_size = 512
val_split = 0.2
test_split = 0.5

# Set model parameters (found in hparams.yaml file)
hidden_dim = 400
dropout = 0.4
layers = 4
heads = 4

# Set up dataset and Datamodule
dataset = SpectrumDataset()
dataloader = SpectrumDataModule(dataset=dataset, batch_size=batch_size, test_split=test_split, val_split=val_split)

max_word_len = dataset.pad_to_len
num_classes = len(dataset.symbol_to_idx)
input_dim = dataset.spectra.shape[-1]
padding_idx = dataset.symbol_to_idx['[nop]']

# define the model
model=Transformer.load_from_checkpoint("./checkpoints/Transformer-epoch=96-step=30070.ckpt")

# Predict with model
trainer = pl.Trainer(gpus=-1)
predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
predicition_targets = []
for _, batch in enumerate(dataloader.predict_dataloader()):
    _,_, targs = batch
    predicition_targets.append(targs)
predicition_targets = np.concatenate(predicition_targets)
acc=0.0
formula_acc = 0.0

for i in tqdm(range(0, 19809)):
    target = predicition_targets[i]
    pred = np.argmax([predictions[i]], axis=2)
    pred_selfies = sf.encoding_to_selfies(pred[0],
                                          vocab_itos=dataset.idx_to_symbol, enc_type='label').split('[nop]')[0]
    targ_selfies = sf.encoding_to_selfies(target, vocab_itos=dataset.idx_to_symbol,
                                          enc_type='label').split('[nop]')[0]
    # Convert to SMILES
    pred_smiles = sf.decoder(pred_selfies)
    targ_smiles = sf.decoder(targ_selfies)

    pred_mol = Chem.MolFromSmiles(pred_smiles)
    targ_mol = Chem.MolFromSmiles(targ_smiles)

    pred_formula = rdMolDescriptors.CalcMolFormula(pred_mol)
    targ_formula = rdMolDescriptors.CalcMolFormula(targ_mol)
    if pred_formula == targ_formula:
        formula_acc +=1
    if targ_smiles==pred_smiles or targ_selfies==pred_selfies:
        acc += 1
    print("pred: ", pred_formula)
    print("targ: ", targ_formula)

print(formula_acc/19809)
print(acc/19809)
