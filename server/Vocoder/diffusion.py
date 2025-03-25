import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import librosa
import soundfile as sf
import time

from .src.diffusion.model.model import Model
from .src.diffusion.model.vocoder.vocoder import Vocoder as HifiGAN
from .src.diffusion.infer_utils import AttrDict, compute_hyperparams_given_schedule, get_eval_noise_schedule, sampling_given_noise_schedule_ddim, MAX_WAV_VALUE


class Vocoder(nn.Module):
    """ Wrapper class """
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        with open(config["vocoder_cfg"]) as f:
            data = f.read()
            self.vocoder_cfg = AttrDict(json.loads(data))
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

        self.model = Model(self.vocoder_cfg).to(self.device)
        
        # load
        state_dict = self.load_checkpoint(config["model_path"], self.device)
        self.model.load_state_dict(state_dict['model'])
        self.model.eval().remove_weight_norm()

        # load other
        self.src_diffusion_root = f"{os.path.dirname(__file__)}/src/diffusion"
        self.load_hifigan()
        self.load_ref_audio()

        # schedule
        self.train_n_sch = torch.linspace(
            float(self.vocoder_cfg.beta_0),
            float(self.vocoder_cfg.beta_T),
            int(self.vocoder_cfg.T)
        ).to(self.device)
        self.dh = compute_hyperparams_given_schedule(self.train_n_sch)
        self.eval_n_sch = get_eval_noise_schedule(self.vocoder_cfg.N, self.dh, self.device)
    
    def load_checkpoint(self, filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    def load_hifigan(self):
        self.hifigan = HifiGAN(
            cfg=f'{self.src_diffusion_root}/model/vocoder/config.json',
            ckpt=f'{self.src_diffusion_root}/ckpt/g_02500000'
        ).to(self.device)

    def load_ref_audio(self):
        self.ref_audio_path = f'{self.src_diffusion_root}/ref_audios/ref.wav'
        audio, sampling_rate = sf.read(self.ref_audio_path)
        audio = audio / MAX_WAV_VALUE
        audio = librosa.util.normalize(audio) * 0.95
        if sampling_rate != self.vocoder_cfg.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.vocoder_cfg.sampling_rate)
        self.ref_audio = torch.FloatTensor(audio)

    def sampling_rate(self) -> int:
        return self.vocoder_cfg.sampling_rate
    
    def forward(self, codes, speaker_id):
        # print(codes)
        if isinstance(codes, str):
            codes = [int(c) for c in codes.strip(' ').split()]
        if isinstance(codes[0], str):
            codes = [int(c) for c in codes]

        assert len(codes) > 0
        codes = torch.LongTensor(codes).repeat_interleave(2)
        
        # km = "310 310 112 112 237 237 411 411 197 197 390 390 121 121 171 171 197 197 492 492 492 492 20 20 20 20 119 119 428 428 189 189 157 157 15 15 153 153 353 353 378 378 116 116 374 374 88 88 498 498 204 204 310 310 157 157 72 72 498 498 189 189 73 73 411 411 134 134 498 498 316 316 498 498 299 299 299 299 498 498 498 498 498 498 498 498 243 243 268 268 362 362 335 335 164 164 21 21 498 498 242 242 493 493 223 223 423 423 498 498 104 104 419 419 193 193 281 281 498 498 92 92 498 498 250 250 239 239 498 498 498 498 498 498 266 266 169 169 243 243 293 293 498 498 168 168 49 49 498 498 293 293 498 498 399 399 303 303 377 377 408 408 424 424 294 294 297 297 341 341 436 436 217 217 390 390 114 114 395 395 450 450 164 164 21 21 192 192 363 363 493 493 201 201 210 210 153 153 3 3 164 164 498 498 362 362 457 457 498 498 8 8 424 424 315 315 391 391 498 498 498 498 112 112 351 351 21 21 498 498 405 405 498 498 334 334 292 292 498 498 116 116 499 499 204 204 53 53 11 11 311 311 498 498 395 395 368 368 498 498"
        
        # pad ref audio
        align = len(codes) * self.vocoder_cfg.sampling_rate // self.vocoder_cfg.km_rate
        if len(self.ref_audio) >= align:
            ref_audio = self.ref_audio[:align]
        elif len(self.ref_audio) < align:
            ref_audio = torch.nn.functional.pad(self.ref_audio, (0, align - len(self.ref_audio)), 'constant')
        assert len(ref_audio) == align

        with torch.no_grad():
        # with torch.inference_mode():
            start = time.time()
            codes = codes.unsqueeze(0).to(self.device)
            ref_audio = ref_audio.unsqueeze(0).to(self.device)
            out = sampling_given_noise_schedule_ddim(self.vocoder_cfg, self.eval_n_sch, self.model, codes, ref_audio)
            # print(out.shape)
            print("DDIM: ", time.time() - start)
            out = self.hifigan(out)
            wav = out[0].squeeze()
            print("HifiGAN: ", time.time() - start)

            return wav
