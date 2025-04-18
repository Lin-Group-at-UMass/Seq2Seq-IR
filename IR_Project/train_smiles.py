from src.models import LSTM_autoregressive, GRU_test, GPT, Transformer
from src.tool_functions import same_smi
import pytorch_lightning as pl
import numpy as np
from dataset_utils_smiles import SpectrumDataset, SpectrumDataModule
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import numpy as np
import selfies as sf
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence model from spectra")
    parser.add_argument('--model', type=str, default='Transformer', choices=['LSTM', 'GRU','GPT', 'Transformer'])
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--calculate_prediction', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=78438379)
    return parser.parse_args()

def get_model(args, max_word_len, num_classes, input_dim, padding_idx):
    model_map = {
        'GRU': GRU_test,
        'LSTM': LSTM_autoregressive,
        'GPT': GPT,
        'Transformer': Transformer
    }
    model_cls = model_map[args.model]
    if args.model in ['Transformer', 'GPT']:
        return model_cls(hidden_dim=args.hidden_dim,
                         dropout=args.dropout,
                         layers=args.layers,
                         max_word_len=max_word_len,
                         num_classes=num_classes,
                         input_dim=input_dim,
                         heads=args.heads,
                         padding_idx=padding_idx)
    else:
        return model_cls(hidden_dim=args.hidden_dim,
                         dropout=args.dropout,
                         layers=args.layers,
                         max_word_len=max_word_len,
                         num_classes=num_classes,
                         input_dim=input_dim,
                         padding_idx=padding_idx)

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
    model = get_model(args, max_word_len, num_classes, input_dim, padding_idx)

    # Trainer
    checkpoint_callback = get_callbacks(args)
    if args.use_gpu and args.model=='Transformer':
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=1, callbacks=checkpoint_callback,
                             accelerator='ddp', precision=16, accumulate_grad_batches=1,
                             gradient_clip_val=0.5,)
    elif args.use_gpu and args.model!='Transformer':
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=1, callbacks=checkpoint_callback,
                             accelerator='ddp', precision=16, accumulate_grad_batches=1,
                             )
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=checkpoint_callback)
    trainer.fit(model, datamodule=dataloader)

    if args.calculate_prediction:
        if args.use_gpu:
            prediction_trainer = pl.Trainer(gpus=1, accelerator='ddp', precision=16,)
        else:
            prediction_trainer = pl.Trainer()
        get_results(model, dataloader, prediction_trainer)
