#!/bin/bash

# Script to test models using test.py
# Usage: ./run_test.sh <model> <checkpoint> [task] [output_file]
# Models:
#   transformer           Train Transformer model with SELFIES encoding dataset
#   lstm                  Train LSTM model with SELFIES encoding dataset
#   gru                   Train GRU model with SELFIES encoding dataset
#   gpt                   Train GPT model with SELFIES encoding dataset
#   transformer-mixture   Train Transformer model with mixture encoding dataset
#   lstm-mixture          Train LSTM model with mixture encoding dataset
#   gru-mixture           Train GRU model with mixture encoding dataset
#   gpt-mixture           Train GPT model with mixture encoding dataset
#   transformer-smiles    Train Transformer model with SMILES encoding dataset
#   lstm-smiles           Train LSTM model with SMILES encoding dataset
#   gru-smiles            Train GRU model with SMILES encoding dataset
#   gpt-smiles            Train GPT model with SMILES encoding dataset
# Tasks:
#   eval                  Evaluate model (default) - activate for selfies, mixture, and smiles
#   generation            Generate molecules - activate for selfies, and mixture
#   topk                  Perform top-k evaluation - activate for selfies only

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model> <checkpoint> [task] [output_file]"
    echo "Task defaults to 'eval' if not specified"
    exit 1
fi

MODEL=$1
CHECKPOINT=$2
TASK=${3:-"eval"}
OUTPUT_FILE=$4

# Handle output file parameter
if [ -n "$OUTPUT_FILE" ]; then
    OUTPUT_ARG="--output_file $OUTPUT_FILE"
else
    OUTPUT_ARG=""
fi

case "$MODEL" in
    transformer)
        echo "Testing Transformer (SELFIES) with task: $TASK..."
        python test.py --model Transformer --mode selfies --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    lstm)
        echo "Testing LSTM (SELFIES) with task: $TASK..."
        python test.py --model LSTM --mode selfies --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gru)
        echo "Testing GRU (SELFIES) with task: $TASK..."
        python test.py --model GRU --mode selfies --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gpt)
        echo "Testing GPT (SELFIES) with task: $TASK..."
        python test.py --model GPT --mode selfies --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    transformer-mixture)
        echo "Testing Transformer (mixture) with task: $TASK..."
        python test.py --model Transformer --mode mixture --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    lstm-mixture)
        echo "Testing LSTM (mixture) with task: $TASK..."
        python test.py --model LSTM --mode mixture --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gru-mixture)
        echo "Testing GRU (mixture) with task: $TASK..."
        python test.py --model GRU --mode mixture --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gpt-mixture)
        echo "Testing GPT (mixture) with task: $TASK..."
        python test.py --model GPT --mode mixture --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    transformer-smiles)
        echo "Testing Transformer (SMILES) with task: $TASK..."
        python test.py --model Transformer --mode smiles --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    lstm-smiles)
        echo "Testing LSTM (SMILES) with task: $TASK..."
        python test.py --model LSTM --mode smiles --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gru-smiles)
        echo "Testing GRU (SMILES) with task: $TASK..."
        python test.py --model GRU --mode smiles --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    gpt-smiles)
        echo "Testing GPT (SMILES) with task: $TASK..."
        python test.py --model GPT --mode smiles --task $TASK \
            --checkpoints "$CHECKPOINT" --batch_size 256 --use_gpu True $OUTPUT_ARG
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Usage: $0 {transformer|lstm|gru|gpt|transformer-mixture|lstm-mixture|gru-mixture|gpt-mixture|transformer-smiles|lstm-smiles|gru-smiles|gpt-smiles} <checkpoint> [task] [output_file]"
        exit 1
        ;;
esac

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo "Test completed successfully."
    if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Results saved to $OUTPUT_FILE"
    fi
else
    echo "Test failed with error code $?"
fi
