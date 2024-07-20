#!/bin/bash -e

apt update
apt install ffmpeg portaudio19-dev -y

## Install all the other dependencies normally
pip install --no-cache-dir -r ml-requirements.txt

## force update huggingface_hub (tokenizers 0.14.1 spuriously require and ancient <=0.18 version)
pip install -U huggingface_hub tokenizers

huggingface-cli download collabora/whisperspeech t2s-small-en+pl.model s2a-q4-tiny-en+pl.model
huggingface-cli download charactr/vocos-encodec-24khz

mkdir -p ./ml/torch/hub/checkpoints/
curl -L -o ./ml/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
mkdir -p ./ml/whisper-live/
curl -L -o ./ml/whisper-live/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx

python3 -c 'from transformers.utils.hub import move_cache; move_cache()'
