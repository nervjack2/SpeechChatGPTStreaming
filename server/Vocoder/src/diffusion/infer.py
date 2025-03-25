import os
import json
import torch
import argparse
import sys
import soundfile as sf
from model.model import Model
import time
from model.vocoder.vocoder import Vocoder
from utils import scan_checkpoint, load_checkpoint
torch.backends.cudnn.benchmark = True

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from meldataset import mel_spectrogram as mel_fn
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0

def std_normal(size, device):
    '''
    Generate the standard Gaussian variable of a certain size
    '''
    return torch.normal(0, 1, size=size).to(device)


def sampling_given_noise_schedule_ddim(h, eval_n_sch, net, km, ref):
    alpha_infer, beta_infer, sigma_infer, steps_infer = eval_n_sch
    N = len(steps_infer)

    # prepare ref
    ref = mel_fn(ref, h.n_fft, h.num_mels, h.sampling_rate,
                 h.hop_size, h.win_size, h.fmin, h.fmax)
    ref = (ref - h.mel_m) / h.mel_s

    # prepare x_T
    L_mel = int(km.size(1) * h.sampling_rate / h.hop_size / h.km_rate)
    x_t = std_normal((1, h.num_mels, L_mel), km.device)

    # precompute, main net
    c, ref = net.URE(km, ref)

    alpha_prev = torch.cat([alpha_infer[0:1], alpha_infer[:-1]])

    for n in range(N - 1, -1, -1):
        ts = steps_infer[n] * torch.ones((1, 1)).to(km.device)
        eps = net(x_t, ts, c, ref)
        # DDIM update
        x0_t = (x_t - torch.sqrt(1 - alpha_infer[n] ** 2) * eps) / alpha_infer[n]
        if n > 0:
            x_t = alpha_prev[n] * x0_t + torch.sqrt(1 - alpha_prev[n] ** 2) * eps
        else:
            x_t = x0_t
    x = x_t * h.mel_s + h.mel_m

    return x

def compute_hyperparams_given_schedule(beta):
    '''
    Compute diffusion process hyperparameters

    Parameters:
    beta (tensor):  beta schedule

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), beta/alpha (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    '''

    T = len(beta)
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, T):
        alpha[t] *= alpha[t - 1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
    alpha = torch.sqrt(alpha).to(beta.device)

    _dh = {}
    _dh['T'], _dh['beta'], _dh['alpha'] = T, beta, alpha
    return _dh


def map_noise_scale_to_time_step(alpha_infer, alpha):
    if alpha_infer < alpha[-1]:
        return len(alpha) - 1
    if alpha_infer > alpha[0]:
        return 0
    for t in range(len(alpha) - 1):
        if alpha[t + 1] <= alpha_infer <= alpha[t]:
            step_diff = alpha[t] - alpha_infer
            step_diff /= alpha[t] - alpha[t + 1]
            return t + step_diff.item()
    return -1


def get_eval_noise_schedule(N, dh, device):
    if N == 1000:
        noise_schedule = torch.linspace(0.000001, 0.01, 1000)
    elif N == 200:
        noise_schedule = torch.linspace(0.0001, 0.02, 200)
    elif N == 50:
        noise_schedule = torch.linspace(0.0001, 0.05, 50)
    elif N == 8:
        noise_schedule = torch.FloatTensor([
            6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
            0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5
        ])
    elif N == 6:
        noise_schedule = torch.FloatTensor([
            1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
            0.006634317338466644, 0.09357017278671265, 0.6000000238418579
        ])
    elif N == 4:
        noise_schedule = torch.FloatTensor([
            3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01
        ])
    elif N == 3:
        noise_schedule = torch.FloatTensor([
            9.0000e-05, 9.0000e-03, 6.0000e-01
        ])
    else:
        raise NotImplementedError

    T, alpha = dh['T'], dh['alpha']
    assert len(alpha) == T

    beta_infer = noise_schedule.to(device)
    N = len(beta_infer)
    alpha_infer = 1 - beta_infer
    sigma_infer = beta_infer + 0
    for n in range(1, N):
        alpha_infer[n] *= alpha_infer[n - 1]
        sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
    alpha_infer = torch.sqrt(alpha_infer)
    sigma_infer = torch.sqrt(sigma_infer)

    # mapping noise scales to time steps
    steps_infer = []
    for n in range(N):
        step = map_noise_scale_to_time_step(alpha_infer[n], alpha)
        if step >= 0:
            steps_infer.append(step)
    steps_infer = torch.FloatTensor(steps_infer).to(device)

    return alpha_infer, beta_infer, sigma_infer, steps_infer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def infer_demo(a, h):
    device = torch.device('cuda')
    model = Model(h).to(device)
    vocoder = Vocoder().to(device)

    train_n_sch = torch.linspace(float(h.beta_0), float(h.beta_T), int(h.T)).to(device)
    dh = compute_hyperparams_given_schedule(train_n_sch)
    eval_n_sch = get_eval_noise_schedule(h.N, dh, device)

    cp = scan_checkpoint(a.cp_dir, 'm_') if a.cp_pth is None else a.cp_pth

    state_dict = load_checkpoint(cp, device)
    model.load_state_dict(state_dict['model'])

    model.eval().remove_weight_norm()

    # TODO: read units here
    # TODO: You have to duplicate the tokens!!
    km = "310 310 112 112 237 237 411 411 197 197 390 390 121 121 171 171 197 197 492 492 492 492 20 20 20 20 119 119 428 428 189 189 157 157 15 15 153 153 353 353 378 378 116 116 374 374 88 88 498 498 204 204 310 310 157 157 72 72 498 498 189 189 73 73 411 411 134 134 498 498 316 316 498 498 299 299 299 299 498 498 498 498 498 498 498 498 243 243 268 268 362 362 335 335 164 164 21 21 498 498 242 242 493 493 223 223 423 423 498 498 104 104 419 419 193 193 281 281 498 498 92 92 498 498 250 250 239 239 498 498 498 498 498 498 266 266 169 169 243 243 293 293 498 498 168 168 49 49 498 498 293 293 498 498 399 399 303 303 377 377 408 408 424 424 294 294 297 297 341 341 436 436 217 217 390 390 114 114 395 395 450 450 164 164 21 21 192 192 363 363 493 493 201 201 210 210 153 153 3 3 164 164 498 498 362 362 457 457 498 498 8 8 424 424 315 315 391 391 498 498 498 498 112 112 351 351 21 21 498 498 405 405 498 498 334 334 292 292 498 498 116 116 499 499 204 204 53 53 11 11 311 311 498 498 395 395 368 368 498 498"
    km = [int(k) for k in km.split()]
    km = torch.LongTensor(km)

    audio, sampling_rate = sf.read(a.ref_audio)
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    if sampling_rate != h.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=h.sampling_rate)

    audio = torch.FloatTensor(audio)
    align = len(km) * h.sampling_rate // h.km_rate
    if len(audio) > align:
        audio = audio[:align]
    elif len(audio) < align:
        audio = torch.nn.functional.pad(audio, (0, align - len(audio)), 'constant')
    assert len(audio) == align
    km = km.unsqueeze(0)
    audio = audio.unsqueeze(0)

    # print(audio.shape)
    # print(km.shape)
    with torch.no_grad():
        with torch.inference_mode():
            start = time.time()
            km = km.to(device)
            audio = audio.to(device)
            # from model.util import sampling_given_noise_schedule
            out = sampling_given_noise_schedule_ddim(h, eval_n_sch, model, km, audio)
            # print(out)
            middle = time.time()
            out = vocoder(out)
            end = time.time()

            save_dir = a.out_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            sf.write(os.path.join(save_dir, "reconstructed_audio.wav"), out[0].cpu().numpy().squeeze(), h.sampling_rate)

            print("sampling  : ", round(middle - start, 2))
            print("total     : ", round(end - start, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_dir', default='ckpt')
    parser.add_argument('--cp_pth', default=None)
    parser.add_argument('--ref_audio', default='ref_audios/ref.wav')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='generated_audio')
    parser.add_argument('--config', default='config.json')
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    h = AttrDict(json.loads(data))
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed)

    infer_demo(a, h)


if __name__ == '__main__':
    main()
