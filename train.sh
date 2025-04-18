#!/bin/bash

if [ "$1" == "transformer" ]; then
    echo "Training the Transformer model..."
    python Spectrum2Structure/train.py --model Transformer --hidden_dim 768 --dropout 0.1 --layers 6 --heads 6 --batch_size 256 --max_epochs 95 --lr 1e-4 --weight_decay 1e-5
    
elif [ "$1" == "lstm" ]; then
    echo "Running large model..."
    python Spectrum2Structure/train.py --model LSTM --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gru" ]; then
    echo "Running large model..."
    python Spectrum2Structure/train.py --model GRU --hidden_dim 400 --dropout 0.1 --layers 4 --batch_size 256 --max_epochs 65 --lr 1e-3
    
elif [ "$1" == "gpt" ]; then
    echo "Running large model..."
    python Spectrum2Structure/train.py --model GPT --hidden_dim 400 --dropout 0.1 --layers 4 --heads 4 --batch_size 256 --max_epochs 65 --lr 1e-4 --weight_decay 1e-5

else
    echo "Usage: $0 [lstm|gru|gpt|transformer]"
fi
