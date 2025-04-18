#!/bin/bash

# Script to test models using test.py
# Usage: ./run_test.sh <model> <checkpoint> [task] [output_file]
# Models:
#   transformer           Test Transformer model with SELFIES encoding
#   lstm                  Test LSTM model with SELFIES encoding
#   gru                   Test GRU model with SELFIES encoding
#   gpt                   Test GPT model with SELFIES encoding
#   transformer-mixture   Test Transformer model with mixture encoding
#   lstm-mixture          Test LSTM model with mixture encoding
#   gru-mixture           Test GRU model with mixture encoding
#   gpt-mixture           Test GPT model with mixture encoding
#   transformer-smiles    Test Transformer model with SMILES encoding
#   lstm-smiles           Test LSTM model with SMILES encoding
#   gru-smiles            Test GRU model with SMILES encoding
#   gpt-smiles            Test GPT model with SMILES encoding
# Tasks:
#   eval                  Evaluate model (default)
#   generation            Generate molecules
#   topk                  Perform top-k evaluation

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
