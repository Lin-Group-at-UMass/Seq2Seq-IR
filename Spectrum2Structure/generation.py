import sys

import warnings
from models import LSTM,GRU,AttentionLSTM,AttentionGRU,LSTM_autoregressive, test_1 ,LSTM_autoregressive_attention,GPT,Transformer, GRU_test

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
model=GRU_test.load_from_checkpoint("./checkpoints/GRU-epoch=64-step=20150.ckpt")

#
# # 定义checkpoint
# checkpoint_callback = ModelCheckpoint(
#     monitor='step',
#     dirpath='checkpoints/',
#     filename='GPT-{epoch:02d}-{step:02d}',
#     save_top_k=3,
#     mode='max',
# )
#
# # 定义训练器并进行训练
# trainer = pl.Trainer(
#     max_epochs=80,  # 这个可以根据实际情况调整
#     gpus=-1,
#     #gpus=2, accelerator='ddp', precision=16, accumulate_grad_batches=1,
#     #gradient_clip_val=0.5,
#     callbacks=[checkpoint_callback],
# )
#
# trainer.fit(model, datamodule=dataloader)

# Predict with model
trainer = pl.Trainer(gpus=1, accelerator='ddp', precision=16)
predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
predicition_targets = []
for _, batch in enumerate(dataloader.predict_dataloader()):
    _,_, targs = batch
    predicition_targets.append(targs)
predicition_targets = np.concatenate(predicition_targets)

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)

def get_fingerprint_similarity(pred_mol, targ_mol, metric=DataStructs.CosineSimilarity):
    pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, useFeatures=True)
    targ = AllChem.GetMorganFingerprintAsBitVect(targ_mol, 2, useFeatures=True)
    return DataStructs.FingerprintSimilarity(pred, targ, metric=metric)

def get_bemis_murcko_smiles(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core)

molecular_accuracy = 0
scaffold_accuracy = 0
fp_sim = 0
from rdkit.Chem import Descriptors, rdMolDescriptors
import csv
formula_acc = 0.0
log_path = "result_GRU.csv"
with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["pred_smiles", "targ_smiles", "pred_selfies", "targ_selfies", "pred_inchi", "targ_inchi"])
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

    print("pred: ", pred_smiles)
    print("targ: ", targ_smiles)

    if pred_selfies == targ_selfies or pred_smiles == targ_smiles or same_smi(pred_smiles, targ_smiles):
        molecular_accuracy += 1
# Convert to RDKit Mol
    pred_mol = smiles_to_mol(pred_smiles)
    targ_mol = smiles_to_mol(targ_smiles)

    pred_formula = rdMolDescriptors.CalcMolFormula(pred_mol)
    targ_formula = rdMolDescriptors.CalcMolFormula(targ_mol)
    if pred_formula == targ_formula:
        formula_acc +=1

    # Get fingerprint similarity
    fp_sim += get_fingerprint_similarity(pred_mol, targ_mol)

    # Get Murcko Scaffold SMILES
    pred_scaffold = get_bemis_murcko_smiles(pred_mol)
    targ_scaffold = get_bemis_murcko_smiles(targ_mol)

    # Check scaffold accuracy
    if pred_scaffold == targ_scaffold:
        scaffold_accuracy += 1
    pred_inchi = get_InchiKey(pred_smiles)
    targ_inchi = get_InchiKey(targ_smiles)
    with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([pred_smiles,targ_smiles,pred_selfies,targ_selfies,pred_inchi,targ_inchi])

# Get average similarity
fp_sim = fp_sim / 19809


# Get % of correct molecules
molecular_accuracy = molecular_accuracy / 19809


# Get % of correct scaffolds
scaffold_accuracy = scaffold_accuracy / 19809
formula_acc = formula_acc / 19809
print("accuracy:", molecular_accuracy)
print("formula_acc:", formula_acc)
print("average_similarity:", fp_sim)
print("scaffold_accuracy", scaffold_accuracy)