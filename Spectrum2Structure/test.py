import warnings
import csv
import torch
import selfies as sf
import numpy as np
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# Import model classes
from src.models import LSTM_autoregressive, GRU_test, GPT, Transformer
from src.models_mixture import LSTM_Mixture, GRU_Mixture, GPT_Mixture, Transformer_Mixture
from src.models_topk import LSTM_autoregressive_topk, GRU_topk, GPT_topk, Transformer_topk

# Import utility functions
from src.tool_functions import same_smi, get_InchiKey
from src.tool_functions import score as score_func

# Import dataset utilities for different encodings
from dataset_utils import SelfiesSpectrumDataset, SelfiesSpectrumDataModule, \
    MixtureSpectrumDataset, MixtureSpectrumDataModule, \
    SmilesSpectrumDataset, SmilesSpectrumDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Test models trained from spectra")
    parser.add_argument('--mode', type=str, default='selfies', choices=['selfies', 'smiles', 'mixture'],
                        help='Choose encoding mode: selfies, smiles, or mixture')
    parser.add_argument('--task', type=str, default='eval',
                        choices=['eval', 'generation', 'topk'],
                        help='Task type: evaluation, generation, or top-k prediction')
    parser.add_argument('--model', type=str, default='Transformer',
                        choices=['LSTM', 'GRU', 'GPT', 'Transformer'],
                        help='Model architecture')
    parser.add_argument('--checkpoints', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Whether to use GPU for inference')
    parser.add_argument('--seed', type=int, default=78438379,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Custom output file name (optional)')
    return parser.parse_args()


def get_model(args):
    # Select model class based on mode and task
    if args.mode == 'selfies' and (args.task =='eval' or args.task =='generation' or args.task =='topk'):
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
    elif args.mode == 'smiles' and args.task =='eval':
        model_map = {
            'GRU': GRU_test,
            'LSTM': LSTM_autoregressive,
            'GPT': GPT,
            'Transformer': Transformer
        }
    elif args.mode == 'mixture' and (args.task =='eval' or args.task =='generation'):
        model_map = {
            'GRU': GRU_Mixture,
            'LSTM': LSTM_Mixture,
            'GPT': GPT_Mixture,
            'Transformer': Transformer_Mixture
        }
    else:
        raise ValueError(f"Unknown {args.mode} mode for the {args.task} task.")

    model_cls = model_map[args.model]
    model = model_cls.load_from_checkpoint(args.checkpoints)

    return model


def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        monitor='step',
        dirpath='checkpoints/',
        filename=f"{args.mode}_{args.model}" + '-{epoch:02d}-{step:02d}',
        save_top_k=3,
        mode='max')
    return checkpoint_callback


def get_output_filename(args):
    """Generate output filename based on args"""
    if args.output_file:
        return args.output_file

    if args.task == 'generation':
        return f"result_{args.mode}_{args.model}.csv"
    elif args.task == 'topk':
        return f"{args.mode}_{args.model}_topk_output.xlsx"
    else:
        return None


def get_results_selfies(model, dataloader, trainer, dataset, num_samples=None):
    """Evaluation for SELFIES encoding"""
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    prediction_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, _, targs = batch
        prediction_targets.append(targs)
    prediction_targets = np.concatenate(prediction_targets)

    if num_samples is None:
        num_samples = len(prediction_targets)

    molecular_accuracy = 0

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


def get_results_smiles(model, dataloader, trainer, dataset, num_samples=None):
    """Evaluation for SMILES encoding"""
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    prediction_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, inputs, targets = batch
        prediction_targets.append(targets.cpu())
    prediction_targets = np.concatenate(prediction_targets)

    if num_samples is None:
        num_samples = len(prediction_targets)

    molecular_accuracy = 0

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


def get_generation_selfies(model, dataloader, trainer, dataset, output_file, num_samples=None):
    """Generation for SELFIES encoding"""
    predictions = np.concatenate(trainer.predict(model, dataloaders=dataloader))
    prediction_targets = []
    for _, batch in enumerate(dataloader.predict_dataloader()):
        _, _, targs = batch
        prediction_targets.append(targs)
    prediction_targets = np.concatenate(prediction_targets)

    if num_samples is None:
        num_samples = len(prediction_targets)

    molecular_accuracy = 0

    # Create output file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["pred_smiles", "targ_smiles", "pred_selfies", "targ_selfies", "pred_inchi", "targ_inchi"])

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

        # Get InChI keys
        pred_inchi = get_InchiKey(pred_smiles)
        targ_inchi = get_InchiKey(targ_smiles)

        # Write to output file
        with open(output_file, 'a', newline='', encoding='utf-8') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow([pred_smiles, targ_smiles, pred_selfies, targ_selfies, pred_inchi, targ_inchi])

    # Get % of correct molecules
    molecular_accuracy = molecular_accuracy / num_samples
    print("accuracy:", molecular_accuracy)
    print("Generation complete. Results saved to", output_file)
    return molecular_accuracy


def get_topk_results(model, dataloader, trainer, dataset, output_file):
    """Top-k evaluation"""
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
                                                      vocab_itos=dataset.idx_to_symbol, enc_type='label').split(
                    '[nop]')[0]
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
    df_scores.to_excel(output_file, index=False)
    print("Topk Generation complete. Results saved to", output_file)
    return df_scores


def main():
    args = parse_args()

    # Set seed and deterministic behavior
    seed = args.seed
    deterministic = True
    seed_everything(seed, workers=deterministic)
    torch.use_deterministic_algorithms(deterministic)

    # Check valid combinations
    if args.mode == 'mixture' and args.task == 'topk':
        raise ValueError("The 'mixture' mode does not support 'topk' task")
    if args.mode == 'smiles' and args.task != 'eval':
        raise ValueError("The 'smiles' mode only supports 'eval' task currently")

    # Setup datasets based on mode
    if args.mode == 'selfies':
        dataset = SelfiesSpectrumDataset()
        dataloader = SelfiesSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5, 
                                               val_split=0.2)

    elif args.mode == 'smiles':
        dataset = SmilesSpectrumDataset()
        dataloader = SmilesSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5, 
                                              val_split=0.2)

    elif args.mode == 'mixture':
        dataset = MixtureSpectrumDataset()
        dataloader = MixtureSpectrumDataModule(dataset=dataset, batch_size=args.batch_size, test_split=0.5, 
                                               val_split=0.2)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Determine number of samples
    num_samples = len(dataset) if hasattr(dataset, '__len__') else 19809  # Default value

    # Load model
    model = get_model(args)

    # Setup trainer
    trainer_kwargs = {}
    if args.use_gpu:
        trainer_kwargs.update({
            'gpus': 1,
            'accelerator': 'ddp',
            'precision': 16
        })

    prediction_trainer = pl.Trainer(**trainer_kwargs)

    # Get output file name
    output_file = get_output_filename(args)

    # Execute task
    if args.task == 'eval':
        if args.mode == 'selfies' or args.mode == 'mixture':
            accuracy = get_results_selfies(model, dataloader, prediction_trainer, dataset, num_samples)
        elif args.mode == 'smiles':
            accuracy = get_results_smiles(model, dataloader, prediction_trainer, dataset, num_samples)
        print(f"Final {args.mode} accuracy with {args.model} model: {accuracy:.4f}")

    elif args.task == 'generation':
        if args.mode == 'selfies' or args.mode == 'mixture':
            accuracy = get_generation_selfies(model, dataloader, prediction_trainer, dataset, output_file, num_samples)
        else:
            raise ValueError(f"Generation task not supported for mode: {args.mode}")
        print(f"Final {args.mode} generation accuracy with {args.model} model: {accuracy:.4f}")

    elif args.task == 'topk':
        if args.mode == 'selfies':
            results = get_topk_results(model, dataloader, prediction_trainer, dataset, output_file)
        else:
            raise ValueError(f"Top-k task not supported for mode: {args.mode}")
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == '__main__':
    main()
