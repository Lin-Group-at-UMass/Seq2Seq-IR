import sys
import warnings
from models_topk import GRU_topk, LSTM_autoregressive_topk, GPT_topk, Transformer_topk
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
from ctypes import ArgumentError
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from score import score as score_func

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

# Set up dataset and Datamodule
dataset = SpectrumDataset()
dataloader = SpectrumDataModule(dataset=dataset, batch_size=batch_size, test_split=test_split, val_split=val_split)

max_word_len = dataset.pad_to_len
num_classes = len(dataset.symbol_to_idx)
input_dim = dataset.spectra.shape[-1]
padding_idx = dataset.symbol_to_idx['[nop]']

# define the model
#model=LSTM_topk.load_from_checkpoint("./checkpoints/LSTM_epoch=52-step=8215.ckpt")
model=LSTM_autoregressive_topk.load_from_checkpoint("./checkpoints/LSTM_autoregressive-epoch=64-step=20150.00.ckpt")
model.eval()
# 用于存储生成的候选结果和对应的 ground truth
predictions = []           # 每个元素为一个 batch 的 beam search 结果
prediction_targets = []    # 每个元素为一个 batch 中的 ground truth SMILES 字符串

# 遍历 dataloader 中的 batch（注意：这里使用的是 predict_dataloader）
for i, batch in enumerate(dataloader.predict_dataloader()):
    warnings.warn(f"Batch: {i}", UserWarning)
    # 假设 batch 格式为：(spectra, inputs, targets)
    _, _, targs = batch
    # 这里 beam_width 设为 10，可根据需要调整
    batch_predictions = model.predict_step(batch, batch_idx=0, beam_width=10)
    predictions.append(batch_predictions)
    prediction_targets.append(targs.numpy())

all_pred_smiles = []
all_targ_smiles = []
for batch_idx, batch_pred in enumerate(predictions):
    warnings.warn(f"Batch: {batch_idx}", UserWarning)
    for i, sample_beam in enumerate(batch_pred):
        target = prediction_targets[batch_idx][i]
        targ_selfies = sf.encoding_to_selfies(target, vocab_itos=dataset.idx_to_symbol,
                                              enc_type='label').split('[nop]')[0]
        targ_smiles = sf.decoder(targ_selfies)
        all_targ_smiles.append(targ_smiles)

        candidate_smiles = []
        for candidate in sample_beam:
            seq, score, _, _ = candidate  # 提取 token 序列和累计得分（score 可根据需要保留或忽略）
            seq = np.array(seq[1:])
            pred_selfies = sf.encoding_to_selfies(seq,
                                                  vocab_itos=dataset.idx_to_symbol, enc_type='label').split('[nop]')[0]
            pred_smiles = sf.decoder(pred_selfies)
            smiles = pred_smiles
            candidate_smiles.append(smiles)
        all_pred_smiles.append(candidate_smiles)

# --- 4. 计算得分 --- #
# 调用你之前定义的 score 函数，对比生成的候选 SMILES 与 ground truth
df_scores = score_func(all_pred_smiles, all_targ_smiles)

print("Results: Smiles Match")
print(
    "Top 1: {:.3f}, Top 5: {:.3f}, Top 10: {:.3f}".format(
        df_scores["top1"].sum() / len(df_scores) * 100,
        df_scores["top5"].sum() / len(df_scores) * 100,
        df_scores["top10"].sum() / len(df_scores) * 100,
    )
)

print("Results: Scaffold Match")
print(
    "Top 1 Scaffold: {:.3f}, Top 5 Scaffold: {:.3f}, Top 10 Scaffold: {:.3f}".format(
        df_scores["top1_scaffold"].sum() / len(df_scores) * 100,
        df_scores["top5_scaffold"].sum() / len(df_scores) * 100,
        df_scores["top10_scaffold"].sum() / len(df_scores) * 100,
    )
)

print("Results: InchiKey Match")
print(
    "Top 1 InchiKey: {:.3f}, Top 5 InchiKey: {:.3f}, Top 10 InchiKey: {:.3f}".format(
        df_scores["top1_inchikey"].sum() / len(df_scores) * 100,
        df_scores["top5_inchikey"].sum() / len(df_scores) * 100,
        df_scores["top10_inchikey"].sum() / len(df_scores) * 100,
    )
)

df_scores = df_scores.reset_index().rename(columns={'index': 'tgt_smiles'})
df_scores.to_excel("LSTM_output.xlsx", index=False)
