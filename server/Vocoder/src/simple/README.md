# End-to-end Spoken Llama

## Introduction
This is an easy example of demonstrating how to generate single spoken turn given one user's response. It is not real-time, and still need some acceleration (batch decoding or multi-process).

This model is initialized by Llama3-8B-instruct, and add 2000 speech units for fine-tuning on spoken dialogue tasks, so this model can accept text / speech as input, and it can also generate text / speech.

The whole process is splitted into 3 parts: speech2unit, language modeling, and speech resynthesis

## Download checkpoint
```sh
gdown 1b2niOM0haS_7k_6X4jD405-qC06O00pP
tar zxvf ckpt.tar.gz
```

## Speech2unit
We use text and speech units interleaving sequence to represent speech signals, so we should use HuBERT to quantize waveform, and insert its transcription in between the HuBERT units.
```sh
pip install faster-whisper
# fp16 might result in some difference in generated units compared to fp32
python speech2unit.py \
    --input_audio <audio_path> \
    --output_path <output_path> \
    --fp16
### if you encounter the libcudnn errors ###
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb 
# dpkg -i cuda-keyring_1.0-1_all.deb 
# apt update && apt upgrade
# apt install libcudnn8 libcudnn8-dev
```

## Language modeling

```sh
pytho lm_generate.py \
    --input_txt <output_path_in_speech2unit> \
    --output_path <output_path_for_continuation> \
    --user_modal speech # user's modality \
    --machine_moal speech # model's modality
```

## Speech resynthesis
Resynthesis speech from generated machine response. Although the model outputs interleaving sequence, we only use the HuBERT units to resynthesis speech (note that the content of text and HuBERT tokens might differ sometimes).
```sh
python generate_waveform.py \
    --in_file <output_path_in_language_modeling> \
    --output_path <output_path_for_audio> \
    --spk <speaker_id>
```
