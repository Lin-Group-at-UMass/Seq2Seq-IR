import argparse
import numpy as np
import torch
import selfies as sf
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# Import model classes
from src.models import LSTM_autoregressive, GRU_test, GPT, Transformer
from src.models_mixture import LSTM_Mixture, GRU_Mixture, GPT_Mixture, Transformer_Mixture
from src.tool_functions import same_smi

# Import dataset utilities
from dataset_utils import SelfiesSpectrumDataset, SelfiesSpectrumDataModule, \
    MixtureSpectrumDataset, MixtureSpectrumDataModule, \
    SmilesSpectrumDataset, SmilesSpectrumDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence model from spectra")
    parser.add_argument('--model', type=str, default='Transformer', choices=['LSTM', 'GRU', 'GPT', 'Transformer'])
    parser.add_argument('--mode', type=str, default='selfies', choices=['selfies', 'smiles', 'mixture'])
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--calculate_prediction', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=78438379)
    return parser.parse_args()


def get_model(args, max_word_len, num_classes, input_dim, padding_idx):
    if args.mode == 'selfies' or args.mode == 'smiles':
        model_map = {
            'GRU': GRU_test,
            'LSTM': LSTM_autoregressive,
            'GPT': GPT,
            'Transformer': Transformer
        }
    elif args.mode == 'mixture':
        model_map = {
            'GRU': GRU_Mixture,
            'LSTM': LSTM_Mixture,
            'GPT': GPT_Mixture,
            'Transformer': Transformer_Mixture
        }
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

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
        filename=f"{args.mode}_{args.model}" + '-{epoch:02d}-{step:02d}',
        save_top_k=3,
        mode='max')
    return checkpoint_callback


def get_results_selfies(model, dataloader, trainer, dataset):
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    prediction_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, _, targs = batch
        prediction_targets.append(targs)
    prediction_targets = np.concatenate(prediction_targets)

    molecular_accuracy = 0
    num_samples = len(prediction_targets)

    for i in tqdm(range(num_samples)):
        target = prediction_targets[i]
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

    # Get % of correct molecules
    molecular_accuracy = molecular_accuracy / num_samples
    print("accuracy:", molecular_accuracy)
    return molecular_accuracy


def get_results_smiles(model, dataloader, trainer, dataset):
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    prediction_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, inputs, targets = batch
        prediction_targets.append(targets.cpu())
    prediction_targets = np.concatenate(prediction_targets)

    molecular_accuracy = 0
    num_samples = len(prediction_targets)

    for i in tqdm(range(num_samples)):
        target = prediction_targets[i]
        pred = predictions[i].argmax(-1)
        target = dataset.tokenizer.decode(target).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        print("Targ: ", target)
        pred = dataset.tokenizer.decode(pred).replace("<pad>", "").replace("<s>", "").replace("</s>", "")
        print("Pred: ", pred)
        if same_smi(pred, target):
            molecular_accuracy += 1

    # Get % of correct molecules
    molecular_accuracy = molecular_accuracy / num_samples
    print("accuracy:", molecular_accuracy)
    return molecular_accuracy


def get_results(model, dataloader, trainer, dataset, mode):
    if mode == 'selfies' or mode == 'mixture':
        return get_results_selfies(model, dataloader, trainer, dataset)
    elif mode == 'smiles':
        return get_results_smiles(model, dataloader, trainer, dataset)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    args = parse_args()

    # Set seed and deterministic behavior
    seed = args.seed
    deterministic = True
    seed_everything(seed, workers=deterministic)
    torch.use_deterministic_algorithms(deterministic)

    # Dataset setup based on mode
    if args.mode == 'selfies':
        dataset = SelfiesSpectrumDataset()
        dataloader = SelfiesSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5, val_split=0.2)
        # Meta info
        max_word_len = dataset.pad_to_len
        num_classes = len(dataset.symbol_to_idx)
        input_dim = dataset.spectra.shape[-1]
        padding_idx = dataset.symbol_to_idx['[nop]']
    elif args.mode == 'smiles':
        dataset = SmilesSpectrumDataset()
        dataloader = SmilesSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5,
                                               val_split=0.2)
        # Meta info
        max_word_len = dataset.pad_to_len
        num_classes = 28  # Fixed for SMILES encoder
        input_dim = dataset.spectra.shape[-1]
        padding_idx = dataset.tokenizer.pad_token_id
    elif args.mode == 'mixture':
        dataset = MixtureSpectrumDataset()
        dataloader = MixtureSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5,
                                                val_split=0.2)
        # Meta info
        max_word_len = dataset.pad_to_len
        num_classes = len(dataset.symbol_to_idx)
        input_dim = dataset.spectra.shape[-1]
        padding_idx = dataset.symbol_to_idx['[nop]']
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Model
    model = get_model(args, max_word_len, num_classes, input_dim, padding_idx)

    # Trainer
    checkpoint_callback = get_callbacks(args)
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'callbacks': [checkpoint_callback]
    }

    if args.use_gpu:
        trainer_kwargs.update({
            'gpus': 1,
            'accelerator': 'ddp',
            'precision': 16,
            'accumulate_grad_batches': 1
        })

        if args.model == 'Transformer':
            trainer_kwargs['gradient_clip_val'] = 0.5

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dataloader)

    # Calculate prediction accuracy if requested
    if args.calculate_prediction:
        prediction_trainer_kwargs = {}
        if args.use_gpu:
            prediction_trainer_kwargs.update({
                'gpus': 1,
                'accelerator': 'ddp',
                'precision': 16
            })

        prediction_trainer = pl.Trainer(**prediction_trainer_kwargs)
        accuracy = get_results(model, dataloader, prediction_trainer, dataset, args.mode)
        print(f"Final {args.mode} accuracy with {args.model} model: {accuracy:.4f}")


if __name__ == '__main__':
    main()
