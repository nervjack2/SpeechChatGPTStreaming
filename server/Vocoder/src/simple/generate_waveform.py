# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import argparse
import json
import logging
import os
from pathlib import Path
import soundfile as sf
import torch

from tqdm import tqdm
from vocoder.models import CodeGenerator
from vocoder.utils import AttrDict
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, data, sample_id, pred_wav):
    sf.write(
        args.output_path,
        pred_wav.detach().cpu().numpy(),
        args.sample_rate,
    )


def load_data(in_file):
    with open(in_file) as f:
        data = [ast.literal_eval(line.strip()) for line in f]

    return data

def code2wav(vocoder, codes, speaker_id, use_cuda=True):
    if isinstance(codes, str):
        codes = [int(c) for c in codes.strip(' ').split()]

    if len(codes) > 0:
        inp = dict()
        inp["code"] = torch.LongTensor(codes).view(1, -1)
        inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1) 
        if use_cuda:
            inp = {k: v.cuda() for k, v in inp.items()}
        wav = vocoder(**inp).squeeze()
    return wav
    
def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict

def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available()

    with open(args.vocoder_cfg) as f:
        vocoder_cfg = AttrDict(json.load(f))
    vocoder = CodeGenerator(vocoder_cfg)
    state_dict_g = load_checkpoint(args.vocoder)
    vocoder.load_state_dict(state_dict_g['generator'])
    if use_cuda:
        vocoder = vocoder.cuda()

    data = load_data(args.in_file)

    for i, d in tqdm(enumerate(data), total=len(data)):
        wav = code2wav(vocoder, d["code"], args.spk, use_cuda=use_cuda)
        if wav is not None:
            dump_result(args, d, i, wav)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        type=str,
        required=True,
        help="Input file following the same format of the output from sample.py ('f0' and 'cpc_km100/hubert' are required fields)",
    )
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg",
        type=str,
        required=True,
        help="path to the vocoder config",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--output_path",
        type=str,
        default="tmp.wav",
        help="Output path",
    )
    parser.add_argument("--spk", default=4, type=int)

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
