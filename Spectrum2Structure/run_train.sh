#!/bin/bash

# Script to train models using train.py
# Usage: ./run_train.sh <model>
# Models:
#   transformer           Train Transformer model with SELFIES encoding
#   lstm                  Train LSTM model with SELFIES encoding
#   gru                   Train GRU model with SELFIES encoding
#   gpt                   Train GPT model with SELFIES encoding
#   transformer-mixture   Train Transformer model with mixture encoding
#   lstm-mixture          Train LSTM model with mixture encoding
#   gru-mixture           Train GRU model with mixture encoding
#   gpt-mixture           Train GPT model with mixture encoding
#   transformer-smiles    Train Transformer model with SMILES encoding
#   lstm-smiles           Train LSTM model with SMILES encoding
#   gru-smiles            Train GRU model with SMILES encoding
#   gpt-smiles            Train GPT model with SMILES encoding

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

case "$1" in
    transformer)
        echo "Training Transformer (SELFIES)..."
        python train.py --model Transformer --mode selfies \
            --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 \
            --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5
        ;;
    lstm)
        echo "Training LSTM (SELFIES)..."
        python train.py --model LSTM --mode selfies \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gru)
        echo "Training GRU (SELFIES)..."
        python train.py --model GRU --mode selfies \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gpt)
        echo "Training GPT (SELFIES)..."
        python train.py --model GPT --mode selfies \
            --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5
        ;;
    transformer-mixture)
        echo "Training Transformer (mixture)..."
        python train.py --model Transformer --mode mixture \
            --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 \
            --batch_size 256 --max_epochs 80 --lr 1e-4 --weight_decay 1e-5
        ;;
    lstm-mixture)
        echo "Training LSTM (mixture)..."
        python train.py --model LSTM --mode mixture \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gru-mixture)
        echo "Training GRU (mixture)..."
        python train.py --model GRU --mode mixture \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gpt-mixture)
        echo "Training GPT (mixture)..."
        python train.py --model GPT --mode mixture \
            --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5
        ;;
    transformer-smiles)
        echo "Training Transformer (SMILES)..."
        python train.py --model Transformer --mode smiles \
            --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 \
            --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5
        ;;
    lstm-smiles)
        echo "Training LSTM (SMILES)..."
        python train.py --model LSTM --mode smiles \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gru-smiles)
        echo "Training GRU (SMILES)..."
        python train.py --model GRU --mode smiles \
            --hidden_dim 400 --dropout 0.1 --layers 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-3
        ;;
    gpt-smiles)
        echo "Training GPT (SMILES)..."
        python train.py --model GPT --mode smiles \
            --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 \
            --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5
        ;;
    *)
        echo "Unknown model: $1"
        echo "Usage: $0 {transformer|lstm|gru|gpt|transformer-mixture|lstm-mixture|gru-mixture|gpt-mixture|transformer-smiles|lstm-smiles|gru-smiles|gpt-smiles}"
        exit 1
        ;;
esac
