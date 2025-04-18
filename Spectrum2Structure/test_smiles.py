from src.models import LSTM_autoregressive, GRU_test, GPT, Transformer
from src.models_topk import LSTM_autoregressive_topk, GRU_topk, GPT_topk, Transformer_topk
from src.tool_functions import same_smi, get_InchiKey
from src.tool_functions import score as score_func
import pytorch_lightning as pl
import numpy as np
from dataset_utils_smiles import SpectrumDataset, SpectrumDataModule
from tqdm import tqdm
import csv
import torch
import selfies as sf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import numpy as np
import selfies as sf
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence model from spectra")
    parser.add_argument('--task', type=str, default='eval', choices=['eval'])
    parser.add_argument('--model', type=str, default='Transformer', choices=['LSTM', 'GRU','GPT', 'Transformer'])
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=78438379)
    return parser.parse_args()

def get_model(args):
    if args.task != 'topk':
        model_map = {
            'GRU': GRU_test,
            'LSTM': LSTM_autoregressive,
            'GPT': GPT,
            'Transformer': Transformer
        }
    else:
        model_map = {
            'GRU': GRU_topk,
            'LSTM': LSTM_autoregressive_topk,
            'GPT': GPT_topk,
            'Transformer': Transformer_topk
        }
    model_cls = model_map[args.model]
    model = model_cls.load_from_checkpoint(args.checkpoints)
    
    return model

def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        monitor='step',
        dirpath='checkpoints/',
        filename=args.model + '-{epoch:02d}-{step:02d}',
        save_top_k=3,
        mode='max',)
    return checkpoint_callback

def get_results(model, dataloader, trainer):
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    predicition_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, inputs, targets = batch
        predicition_targets.append(targets.cpu())
    predicition_targets = np.concatenate(predicition_targets)

    molecular_accuracy = 0

    for i in tqdm(range(0, 19809)):
        target = predicition_targets[i]
        pred = predictions[i].argmax(-1)
        target = dataset.tokenizer.decode(target).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        print("Targ: ",target)
        pred = dataset.tokenizer.decode(pred).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        print("Pred: ",pred)
        if same_smi(pred, target):
            molecular_accuracy+=1

    # Get % of correct molecules
    molecular_accuracy = molecular_accuracy / 19809
    print("accuracy:", molecular_accuracy)

if __name__ == '__main__':
    args = parse_args()

    # Set seed and deterministic behavior
    seed = 78438379
    deterministic = True
    seed_everything(seed, workers=deterministic)
    torch.use_deterministic_algorithms(deterministic)

    # Dataset
    dataset = SpectrumDataset()
    dataloader = SpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5, val_split=0.2)
    
    # Meta info
    max_word_len = dataset.pad_to_len
    num_classes = 28
    input_dim = dataset.spectra.shape[-1]
    padding_idx = dataset.tokenizer.pad_token_id

    # Model
    model = get_model(args)

    # Trainer
    if args.use_gpu:
        prediction_trainer = pl.Trainer(gpus=1, accelerator='ddp', precision=16,)
    else:
        prediction_trainer = pl.Trainer()

    if args.task == 'eval':
        get_results(model, dataloader, prediction_trainer)
    else:
        raise ValueError("Invalid task. Choose from ['eval', 'generation', 'topk']")