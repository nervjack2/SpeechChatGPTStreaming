import socket
import asyncio
import os
import sys
import numpy as np
import pickle
import torch
import joblib
from .. import Define
from .streaming_hubert import StreamingHubertEncoder, ApplyKmeans
from ..utils import length_prefixing, recv_with_length_prefixing, handle_asyncio_exception

class Processor(object):

    SAMPLING_RATE = 16000

    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr
        self.audio_chunks = []
        self.first_chunks = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.min_chunk = config["min-chunk-size"]
        self.max_buffer = config["max-buffer-size"]
        self.km_model = config["km_model"]
        self.encoder = StreamingHubertEncoder()
        self.apply_kmeans = ApplyKmeans(self.km_model, use_gpu=True)

    def reset_buffer(self):
        self.audio_chunks = []
        self.first_chunks = []

    def quantize(self, audio, cur_data_len):
        feat = self.encoder.encode(audio)
        feat = feat[-int(cur_data_len/320):]
        ssl_units = self.apply_kmeans(feat)
        return [f"<|{p}|>" for p in ssl_units][::Define.UNIT_DOWNSAMPLE_RATE]

    async def input_data(self, res):
        loop = asyncio.get_event_loop()
        # self.audio_chunks.append(res["data"])
        self.first_chunks.append(res["data"])
        cur_chunk_len = sum(len(x) for x in self.first_chunks)
        if cur_chunk_len < self.min_chunk * self.SAMPLING_RATE:
            return
        self.audio_chunks += self.first_chunks
        self.first_chunks = []
        total_chunk_len = sum(len(x) for x in self.audio_chunks)
        concat_audio = np.concatenate(self.audio_chunks)

        while total_chunk_len > (self.max_buffer-self.min_chunk) * self.SAMPLING_RATE:
            x = self.audio_chunks.pop(0) # Discrad first piece of audio 
            total_chunk_len -= len(x)

        o = self.quantize(concat_audio, cur_chunk_len)

        res = {
            "type": "user_unit",
            "data": o,
            "input_timestamp": res["input_timestamp"],
        }

        await loop.sock_sendall(self.client_socket, length_prefixing(res))

    async def process(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                if not res:
                    break
                res = pickle.loads(res)
                if res["type"] == "reset":
                    print("========== reset all state for hubert discrete ==========")
                    self.audio_chunks = []
                    continue
                assert res["type"] == "audio"
                await self.input_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print(f"Connection from {self.addr} closed.")
            self.client_socket.close()

class StreamingHuBERTDiscreteServerProcessor(object):
    def __init__(self, config):
        self.config = config
    
    """ This is only a wrapper class. """
    def __init__(self, config):
        self.config = config
    
    async def process(self, client_socket: socket.socket, addr):  # handle one client connection
        loop = asyncio.get_event_loop()
        processor = Processor(self.config, client_socket, addr)

        # main process loop
        fut = loop.create_task(processor.process())
        fut.add_done_callback(handle_asyncio_exception)