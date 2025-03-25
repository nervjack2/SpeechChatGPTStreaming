import os
import torch
import random
import numpy as np
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def stft(x, n_fft, hop_size, win_size, center=False):
    global hann_window
    if str(x.device) not in hann_window:
        hann_window[str(win_size)+'_'+str(x.device)] = torch.hann_window(win_size).to(x.device)
    x = torch.stft(x, n_fft, hop_length=hop_size, win_length=win_size,
                   window=hann_window[str(win_size)+'_'+str(x.device)], center=center,
                   pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    return x


def istft(x, n_fft, hop_size, win_size, center=False):
    global hann_window
    if str(x.device) not in hann_window:
        hann_window[str(win_size)+'_'+str(x.device)] = torch.hann_window(win_size).to(x.device)
    x = torch.istft(x, n_fft, hop_length=hop_size, win_length=win_size,
                    window=hann_window[str(win_size)+'_'+str(x.device)], center=center,
                    normalized=False, onesided=True, return_complex=False)
    return x


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    mk = '_'.join([str(sampling_rate), str(n_fft), str(num_mels), str(fmin), str(fmax), str(y.device)])
    if mk not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[mk] = torch.from_numpy(mel).float().to(y.device)
    if str(win_size)+'_'+str(y.device) not in hann_window:
        hann_window[str(win_size)+'_'+str(y.device)] = torch.hann_window(win_size).to(y.device)

    lpad = (n_fft-hop_size)//2
    y = torch.nn.functional.pad(y.unsqueeze(1), (lpad, n_fft-hop_size-lpad), mode='reflect')
    y = y.squeeze(1)

    spec = stft(y, n_fft, hop_size, win_size, center)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[mk], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def get_dataset_filelist(a, train=True):
    ret = []

    if train: 
        subset = [a.train, a.valid] 
    else: 
        subset = [a.valid]

    for split in subset:
        with open(split+'.km', 'r') as r:
            kms = [l.strip().split(' ') for l in r.readlines()]
        with open(split+'.txt', 'r') as r:
            files = [l.strip() for l in r.readlines()]
        assert len(kms) == len(files)
        ret.append([
                [[int(e) for e in kms[i]],
                os.path.join(f)]
            for i, f in enumerate(files)
        ])
    
    if train:
        return ret[0], ret[1]
    else: 
        return ret[0]


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, files, h, split=True, shuffle=True):
        self.files = files
        self.km_rate = h.km_rate
        self.sampling_rate = h.sampling_rate
        self.pidx = h.num_km
        self.split = split
        random.seed(h.seed)
        if shuffle:
            random.shuffle(self.files)
        self.km_size = h.segment_size*h.km_rate
        self.audio_size = h.segment_size*h.sampling_rate

    def __getitem__(self, index):
        km, wav = self.files[index]

        audio, sampling_rate = load_wav(wav)
        audio = audio/MAX_WAV_VALUE
        audio = normalize(audio)*0.95
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        km = torch.LongTensor(km)
        audio = torch.FloatTensor(audio)
        align_l = len(km)*self.sampling_rate//self.km_rate
        if len(audio) > align_l:
            audio = audio[:align_l]
        elif len(audio) < align_l:
            audio = torch.nn.functional.pad(audio, (0, align_l-len(audio)), 'constant')
        assert len(audio) == align_l

        ref = audio
        if self.split:
            if len(km) >= self.km_size:
                max_s = len(km)-self.km_size
                km_s = random.randint(0, max_s)
                audio_s = km_s*self.sampling_rate//self.km_rate
                ref_s = random.randint(0, max_s)*self.sampling_rate//self.km_rate
                km = km[km_s:km_s+self.km_size]
                ref = audio[ref_s:ref_s+self.audio_size]
                audio = audio[audio_s:audio_s+self.audio_size]
            else:
                km = torch.nn.functional.pad(km, (0, self.km_size-len(km)), 'constant', self.pidx)
                audio = torch.nn.functional.pad(audio, (0, self.audio_size-len(audio)), 'constant')
                ref = audio

        return km, ref, audio

    def __len__(self):
        return len(self.files)
