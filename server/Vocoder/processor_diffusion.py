import socket
import asyncio
import numpy as np
import torch
import time
from threading import Thread
import queue
import pickle
import uuid
import librosa

from server.common.template import DefaultServerProcessor
from .processor_simple import Processor, Vocoder
from . import diffusion


class DiffusionVocoder(Vocoder):
    
    input_queue: queue.Queue
    output_queues: dict[uuid.UUID, queue.Queue]

    def __init__(self, config) -> None:
        self.vocoder = diffusion.Vocoder(config)
        self.input_queue = queue.Queue()
        self.output_queues = {}

        self._warmup()

        self.samples_per_unit = 640

    def forward(self, x: dict):
        """ forward a single chunk of data """
        st = time.time()
        self.n_tokens = 0
        # print(f"Forward: {x}")
        codes = x["message"]
        with torch.no_grad():
            self.n_tokens += len(x["message"])
            audio = self.vocoder.forward(codes=codes, speaker_id=1).detach().cpu().numpy()
            audio = librosa.resample(audio, orig_sr=self.vocoder.sampling_rate(), target_sr=16000)
            audio = librosa.effects.time_stretch(audio, rate=float(len(audio)) / (self.samples_per_unit * self.n_tokens))
            audio = (audio * 32767).astype(np.int16)
        
        tps = self.n_tokens / (time.time()-st)
        print(f"Forward: {time.time()-st:.2f}s. Rate: {int(tps)} tps ({self.n_tokens} tokens). Shape: {len(audio)}.")

        return {"audio": audio}


class VocoderServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        print("Run Stream Diffusion Vocoder on new thread!")
        self.model = DiffusionVocoder(self.config)
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> Processor:
        p = Processor(self.config, client_socket, addr)
        p.connect_model(self.model)
        p.chunker.real_tps = 25  # 25Hz
        return p
