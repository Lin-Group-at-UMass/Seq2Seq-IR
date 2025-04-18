#!/bin/bash

# Single Molecule Training
if [ "$1" == "transformer" ]; then
    echo "Training the Transformer model..."
    python Spectrum2Structure/train.py --model Transformer --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5
    
elif [ "$1" == "lstm" ]; then
    echo "Running LSTM model..."
    python Spectrum2Structure/train.py --model LSTM --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gru" ]; then
    echo "Running GRU model..."
    python Spectrum2Structure/train.py --model GRU --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gpt" ]; then
    echo "Running GPT model..."
    python Spectrum2Structure/train.py --model GPT --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5

# Mixture Molecule Training
elif [ "$1" == "lstm-mixture" ]; then
    echo "Running LSTM-Mixture model..."
    python Spectrum2Structure/train_mixture.py --model LSTM --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gru-mixture" ]; then
    echo "Running GRU-Mixture model..."
    python Spectrum2Structure/train_mixture.py --model GRU --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gpt-mixture" ]; then
    echo "Running GPT-Mixture model..."
    python Spectrum2Structure/train_mixture.py --model GPT --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5

elif [ "$1" == "transformer-mixture" ]; then
    echo "Training the Transformer-Mixture model..."
    python Spectrum2Structure/train_mixture.py --model Transformer --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5

# Single Molecule Training with SMILES format
elif [ "$1" == "lstm-smiles" ]; then
    echo "Running LSTM-Mixture model..."
    python Spectrum2Structure/train_smiles.py --model LSTM --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gru-smiles" ]; then
    echo "Running GRU-Mixture model..."
    python Spectrum2Structure/train_smiles.py --model GRU --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gpt-smiles" ]; then
    echo "Running GPT-Mixture model..."
    python Spectrum2Structure/train_smiles.py --model GPT --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5

elif [ "$1" == "transformer-smiles" ]; then
    echo "Training the Transformer-Mixture model..."
    python Spectrum2Structure/train_smiles.py --model Transformer --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5

else
    echo "Usage: $0 [lstm|gru|gpt|transformer|lstm-mixture|gru-mixture|gpt-mixture|transformer-mixture|lstm-smiles|gru-smiles|gpt-smiles|transformer-smiles]"
fi
