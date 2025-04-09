import sys
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
from models_mixture import LSTM,GRU,Transformer,GPT
import pytorch_lightning as pl
import numpy as np
from dataset_utils_mixture import SpectrumDataset_2, SpectrumDataModule
from tqdm import tqdm
import csv
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
batch_size = 256
val_split = 0.2
test_split = 0.5

# Set model parameters (found in hparams.yaml file)
hidden_dim = 400
dropout = 0.1
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
model=GRU.load_from_checkpoint("./checkpoints/GRU_mixture-epoch=64-step=19305.00.ckpt")

# # ??checkpoint
# checkpoint_callback = ModelCheckpoint(
#     monitor='step',
#     dirpath='checkpoints/',
#     filename='GPT_Mixture-{epoch:02d}-{step:02d}',
#     save_top_k=3,
#     mode='max',
# )
#
# # ??????????
# trainer = pl.Trainer(
#     max_epochs=65,  # ????????????
#     gpus=2, accelerator='ddp', precision=16, accumulate_grad_batches=1,
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

# Calculate accuracy
correct = 0
half_correct = 0
scaffold_correct = 0
scaffold_half_correct = 0
formula_acc = 0.0
half_formula_acc = 0.0
import csv
log_path = "result_GRU_Mixture.csv"
with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["pred_smiles1", "targ_smiles1","pred_smiles2", "targ_smiles2",
                         "pred_selfies1", "targ_selfies1", "pred_selfies2", "targ_selfies2"])

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

    pred_smiles1 = p1
    targ_smiles1 = t1
    pred_smiles2 = p2
    targ_smiles2 = t2
    pred_selfies1 = pred[0]
    targ_selfies1 = targ[0]
    pred_selfies2 = pred[1]
    targ_selfies2 = targ[1]

    with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([pred_smiles1,targ_smiles1,pred_smiles2,targ_smiles2,pred_selfies1,targ_selfies1, pred_selfies2, targ_selfies2])

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

    p1 = rdMolDescriptors.CalcMolFormula(p1)
    p2 = rdMolDescriptors.CalcMolFormula(p2)

    t1 = rdMolDescriptors.CalcMolFormula(t1)
    t2 = rdMolDescriptors.CalcMolFormula(t2)

    if (p1==t1 and p2==t2) or (p1==t2 and p2==t1):
        formula_acc+=1
    if p1==t1 or p2==t2 or p1==t2 or p2==t1:
        half_formula_acc+=1


print(f'''
    Correct:              {correct / 19000 * 100}%
    Half Correct:         {half_correct / 19000 * 100}%
    Scaffold Correct:     {scaffold_correct / 19000 * 100}%
    Scaffold Half Correct:     {scaffold_half_correct / 19000 * 100}%
    formula_acc:     {formula_acc / 19000 * 100}%
    half_formula_acc:     {half_formula_acc / 19000 * 100}%
    ''')