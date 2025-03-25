"""
The very first version of vocoder from YK Fu
"""
import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm

from .src.simple.vocoder.models import CodeGenerator
from .src.simple.vocoder.utils import AttrDict


class Vocoder(nn.Module):
    """ Wrapper class """
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_cuda = torch.cuda.is_available()
        with open(config["vocoder_cfg"]) as f:
            vocoder_cfg = AttrDict(json.load(f))
        self.model = CodeGenerator(vocoder_cfg)
        state_dict_g = self.load_checkpoint(config["model_path"])
        self.model.load_state_dict(state_dict_g['generator'])
        if self.use_cuda:
            self.model.cuda()

    def load_checkpoint(self, filepath):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location='cpu')
        print("Complete.")
        return checkpoint_dict

    def forward(self, codes, speaker_id):
        # print(codes)
        if isinstance(codes, str):
            codes = [int(c) for c in codes.strip(' ').split()]
        if isinstance(codes[0], str):
            codes = [int(c) for c in codes]

        assert len(codes) > 0
        inp = dict()
        inp["code"] = torch.LongTensor(codes).view(1, -1)
        inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1) 
        if self.use_cuda:
            inp = {k: v.cuda() for k, v in inp.items()}
        wav = self.model(**inp).squeeze()
        return wav
