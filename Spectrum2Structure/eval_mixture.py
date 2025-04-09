import warnings
from models_mixture import LSTM,GRU,Transformer,GPT
import pytorch_lightning as pl
import numpy as np
#from dataset_utils import SpectrumDataset, SpectrumDataModule
from dataset_utils_mixture import SpectrumDataset_2, SpectrumDataModule
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
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(m1):
    m1 = MurckoScaffold.GetScaffoldForMol(m1)
    return Chem.MolToInchi(m1)

def judge(key1, key2):
    if key1 is None or key2 is None:
        return False
    return key1 == key2

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

def same_inchi(smi1, smi2):
    key1 = get_InchiKey(smi1)
    key2 = get_InchiKey(smi2)
    if key1 is None or key2 is None:
        return None, None, False
    return key1, key2, judge(key1, key2)

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
dropout = 0.2
layers = 4
heads = 4

# Set up dataset and Datamodule
dataset = SpectrumDataset_2()
dataloader = SpectrumDataModule(dataset=dataset, batch_size=batch_size, test_split=test_split, val_split=val_split)

max_word_len = dataset.pad_to_len
num_classes = len(dataset.symbol_to_idx)
input_dim = dataset.spectra.shape[-1]
padding_idx = dataset.symbol_to_idx['[nop]']

# define the model
model=Transformer(hidden_dim=hidden_dim,
                  dropout=dropout,
                  layers=layers,
                  max_word_len=max_word_len,
                  num_classes=num_classes,
                  input_dim=input_dim,
                  heads=heads,
                  padding_idx=padding_idx).load_from_checkpoint("./checkpoints/Transformer-epoch=62-step=19530.00.ckpt")


# Predict with model
trainer = pl.Trainer(gpus=-1)
predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
predicition_targets = []
for _, batch in enumerate(dataloader.predict_dataloader()):
    _,_, targs = batch
    predicition_targets.append(targs)
predicition_targets = np.concatenate(predicition_targets)

# Calculate accuracy
correct = 0
half_correct = 0
scaffold_correct = 0
scaffold_half_correct = 0

for i in tqdm(range(0, 19000)):
    target = predicition_targets[i]
    pred = np.argmax([predictions[i]], axis=2)
    pred_selfies = sf.encoding_to_selfies(pred[0],
                                          vocab_itos=dataset.idx_to_symbol, enc_type='label')
    targ_selfies = sf.encoding_to_selfies(target, vocab_itos=dataset.idx_to_symbol,
                                          enc_type='label')

    pred = pred_selfies.split('[SPACE]')
    p1 = Chem.MolFromSmiles(sf.decoder(pred[0]))
    p2 = Chem.MolFromSmiles(sf.decoder(pred[1]))

    targ = targ_selfies.split('[SPACE]')
    t1 = Chem.MolFromSmiles(sf.decoder(targ[0]))
    t2 = Chem.MolFromSmiles(sf.decoder(targ[1]))

    if Chem.MolToInchi(p1) == Chem.MolToInchi(t1) and Chem.MolToInchi(p2) == Chem.MolToInchi(t2) or Chem.MolToInchi(
            p1) == Chem.MolToInchi(t2) and Chem.MolToInchi(p2) == Chem.MolToInchi(t1):
        correct += 1

    if get_scaffold(p1) == get_scaffold(t1) and get_scaffold(p2) == get_scaffold(t2) or get_scaffold(
            p1) == get_scaffold(t2) and get_scaffold(p2) == get_scaffold(t1):
        scaffold_correct += 1

    if get_scaffold(p1) == get_scaffold(t1) or get_scaffold(p2) == get_scaffold(t2) or get_scaffold(p1) == get_scaffold(
            t2) or get_scaffold(p2) == get_scaffold(t1):
        scaffold_half_correct += 1

    if Chem.MolToInchi(p1) == Chem.MolToInchi(t1) or Chem.MolToInchi(p2) == Chem.MolToInchi(t2) or Chem.MolToInchi(
            p1) == Chem.MolToInchi(t2) or Chem.MolToInchi(p2) == Chem.MolToInchi(t1):
        half_correct += 1

print(f'''
    Correct:              {correct / 19000 * 100}%
    Half Correct:         {half_correct / 19000 * 100}%
    Scaffold Correct:     {scaffold_correct / 19000 * 100}%
    Scaffold Half Correct:     {scaffold_half_correct / 19000 * 100}%
    ''')