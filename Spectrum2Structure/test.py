import warnings
from src.models import LSTM_autoregressive, GRU_test, GPT, Transformer
from src.models_topk import LSTM_autoregressive_topk, GRU_topk, GPT_topk, Transformer_topk
from src.tool_functions import same_smi, get_InchiKey
from src.tool_functions import score as score_func
import pytorch_lightning as pl
import numpy as np
from dataset_utils import SpectrumDataset, SpectrumDataModule
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
    parser.add_argument('--task', type=str, default='eval', choices=['eval', 'generation', 'topk'])
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
            _,_, targs = batch
            predicition_targets.append(targs)
        predicition_targets = np.concatenate(predicition_targets)

        molecular_accuracy = 0

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

        # Get % of correct molecules
        molecular_accuracy = molecular_accuracy / 19809
        print("accuracy:", molecular_accuracy)

def get_generations(model, dataloader, trainer):
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    predicition_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _,_, targs = batch
        predicition_targets.append(targs)
    predicition_targets = np.concatenate(predicition_targets)

    molecular_accuracy = 0

    log_path = "result_"+args.models+".csv"
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

        pred_inchi = get_InchiKey(pred_smiles)
        targ_inchi = get_InchiKey(targ_smiles)
        with open(log_path, 'a', newline='', encoding='utf-8') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow([pred_smiles,targ_smiles,pred_selfies,targ_selfies,pred_inchi,targ_inchi])

    # Get % of correct molecules
    molecular_accuracy = molecular_accuracy / 19809
    print("accuracy:", molecular_accuracy)
    print("Generation complete. Results saved to", log_path)

def get_topk(model, dataloader, trainer):
    predictions = []           
    prediction_targets = []    

    for i, batch in enumerate(dataloader.predict_dataloader()):
        warnings.warn(f"Batch: {i}", UserWarning)
        _, _, targs = batch
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
                seq, score, _, _ = candidate
                seq = np.array(seq[1:])
                pred_selfies = sf.encoding_to_selfies(seq,
                                                    vocab_itos=dataset.idx_to_symbol, enc_type='label').split('[nop]')[0]
                pred_smiles = sf.decoder(pred_selfies)
                smiles = pred_smiles
                candidate_smiles.append(smiles)
            all_pred_smiles.append(candidate_smiles)

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
    log_path = args.models+"_output.xlsx"
    df_scores.to_excel(log_path, index=False)
    print("Topk Generation complete. Results saved to", log_path)

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
    num_classes = len(dataset.symbol_to_idx)
    input_dim = dataset.spectra.shape[-1]
    padding_idx = dataset.symbol_to_idx['[nop]']

    # Model
    model = get_model(args)

    # Trainer
    if args.use_gpu:
        prediction_trainer = pl.Trainer(gpus=1, accelerator='ddp', precision=16,)
    else:
        prediction_trainer = pl.Trainer()

    if args.task == 'eval':
        get_results(model, dataloader, prediction_trainer)
    elif args.task == 'generation':
        get_generations(model, dataloader, prediction_trainer)
    elif args.task == 'topk':
        get_topk(model, dataloader, prediction_trainer)
    else:
        raise ValueError("Invalid task. Choose from ['eval', 'generation', 'topk']")