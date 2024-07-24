#!/bin/bash -e

test -f /etc/shinit_v2 && source /etc/shinit_v2

echo "Running build-models.sh..."
mkdir -p ./ml/

AUDIO_MODEL=${AUDIO_MODEL:-whisper_small_en}
LLM_MODEL=${LLM_MODEL:-phi-2}
VAD_MODEL=${VAD_MODEL:-whisper-live}

if [ ! -d "./ml/huggingface/models--collabora--whisperspeech" ] || [ -z "$(ls -A ./ml/huggingface/models--collabora--whisperspeech)" ]; then
    echo "collabora/whisperspeech does not exist or is empty. Downloading ..."
    huggingface-cli download collabora/whisperspeech t2s-small-en+pl.model s2a-q4-tiny-en+pl.model --cache-dir ./ml/huggingface
else
    echo "collabora/whisperspeech exists and is not empty. Skipping ..."
fi

if [ ! -d "./ml/huggingface/models--charactr--vocos-encodec-24khz" ] || [ -z "$(ls -A ./ml/huggingface/models--charactr--vocos-encodec-24khz)" ]; then
    echo "charactr/vocos-encodec-24khz does not exist or is empty. Downloading..."
    huggingface-cli download charactr/vocos-encodec-24khz --cache-dir ./ml/huggingface
else
    echo "charactr/vocos-encodec-24khz exists and is not empty. Skipping ..."
fi


if [ ! -d "./ml/$VAD_MODEL" ] || [ -z "$(ls -A ./ml/$VAD_MODEL)" ]; then
    echo "$VAD_MODEL directory does not exist or is empty. Running build-vad.sh..."
    mkdir -p ./ml/$VAD_MODEL/
    curl -L -o ./ml/$VAD_MODEL/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx
else
    echo "$VAD_MODEL directory exists and is not empty. Skipping build-vad.sh..."
fi

if [ ! -d "./ml/torch/hub/checkpoints/" ] || [ -z "$(ls -A ./ml/torch/hub/checkpoints/)" ]; then
    echo "torch/hub/checkpoints/ directory does not exist or is empty. Running build-encodec.sh..."
    mkdir -p ./ml/torch/hub/checkpoints/
    curl -L -o ./ml/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
else
    echo "torch/hub/checkpoints/ directory exists and is not empty. Skipping build-encodec.sh..."
fi


if [ ! -d "./ml/TensorRT-LLM-examples" ] || [ -z "$(ls -A ./ml/TensorRT-LLM-examples)" ]; then
    echo "TensorRT-LLM-examples directory does not exist or is empty. Downloading..."
    git clone -b v0.10.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git ./ml/TensorRT-LLM
    mv ./ml/TensorRT-LLM/examples ./ml/TensorRT-LLM-examples
    rm -rf ./ml/TensorRT-LLM
else
    echo "TensorRT-LLM-examples directory exists and is not empty. Skipping ..."
fi


if [ ! -d "./ml/$AUDIO_MODEL" ] || [ -z "$(ls -A ./ml/$AUDIO_MODEL)" ]; then
    echo "$AUDIO_MODEL directory does not exist or is empty. Running build-whisper.sh..."
    ./scripts/build-whisper.sh
else
    echo "$AUDIO_MODEL directory exists and is not empty. Skipping build-whisper.sh..."
fi

# ./build-mistral.sh

if [ ! -d "./ml/$LLM_MODEL" ] || [ -z "$(ls -A ./ml/$LLM_MODEL)" ]; then
    echo "$LLM_MODEL directory does not exist or is empty. Running build-phi.sh..."
    ./scripts/build-phi.sh $LLM_MODEL
else
    echo "$LLM_MODEL directory exists and is not empty. Skipping build-phi.sh..."
fi
