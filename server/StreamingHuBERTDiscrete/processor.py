import socket
import asyncio
import numpy as np
import pickle
import torch
from threading import Thread
import queue

from .. import Define
from server.common.template import DefaultProcessor, DefaultServerProcessor
from .streaming_hubert import StreamingHubertEncoder, ApplyKmeans


class StreamHubert(object):
    SAMPLING_RATE = 16000
    input_queue: queue.Queue
    output_queue: queue.Queue

    def __init__(self, config):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.config = config
        self.audio_chunks = []
        self.first_chunks = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.min_chunk = config["min-chunk-size"]
        self.max_buffer = config["max-buffer-size"]
        self.km_model = config["km_model"]
        self.encoder = StreamingHubertEncoder()
        self.apply_kmeans = ApplyKmeans(self.km_model, use_gpu=True)
        self._warmup()

    def _warmup(self):
        warmup_audio_len = int(self.max_buffer*self.SAMPLING_RATE)
        warmup_audio = np.random.rand(warmup_audio_len, 1).astype(np.float32)
        o = self.quantize(warmup_audio, warmup_audio_len)
        print("Warmup complete.")

    def reset_buffer(self):
        self.audio_chunks = []
        self.first_chunks = []

    def quantize(self, audio, cur_data_len):
        feat = self.encoder.encode(audio)
        feat = feat[-int(cur_data_len/320):]
        ssl_units = self.apply_kmeans(feat)
        return [f"<|{p}|>" for p in ssl_units][::Define.UNIT_DOWNSAMPLE_RATE]

    # handler functions
    def _handle_reset(self, res):
        self.audio_chunks = []
        print("========== reset all state ==========")

    def _handle_audio(self, res):
        self.first_chunks.append(res["data"])
        cur_chunk_len = sum(len(x) for x in self.first_chunks)
        if cur_chunk_len < self.min_chunk * self.SAMPLING_RATE:
            return
        self.audio_chunks += self.first_chunks
        self.first_chunks = []
        total_chunk_len = sum(len(x) for x in self.audio_chunks)
        concat_audio = np.concatenate(self.audio_chunks)

        while total_chunk_len > (self.max_buffer-self.min_chunk) * self.SAMPLING_RATE:
            x = self.audio_chunks.pop(0) # Discard first piece of audio 
            total_chunk_len -= len(x)

        o = self.quantize(concat_audio, cur_chunk_len)

        res = {
            "type": "user_unit",
            "data": o,
            "input_timestamp": res["input_timestamp"],
        }
        self.output_queue.put(res)

    def run(self):
        """ loop to process input queue """
        while True:
            if self.input_queue.empty():
                continue
            else:
                res = self.input_queue.get()

            # update assistant history
            if res["type"] == "audio":  # update assistant history
                self._handle_audio(res)
            elif res["type"] == "reset":  # reset signal
                self._handle_reset(res)
            else:
                raise NotImplementedError


class StreamingHuBERTDiscreteServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        print("Run Stream Hubert on new thread!")
        self.model = StreamHubert(self.config)
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> DefaultProcessor:
        p = DefaultProcessor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
