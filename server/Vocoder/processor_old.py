import socket
import asyncio
import numpy as np
import torch
import time
from threading import Thread
import queue

from server.common.template import DefaultProcessor, DefaultServerProcessor
from . import simple


class ForwardCriterion(object):
    def __init__(self):
        pass
    
    def exec(self, concat_message, res) -> bool:
        if len(concat_message) == 0:
            return False
        is_breaking = len(concat_message) >= 20 or res.get("eos", False)
        return is_breaking
    
    def reset(self):
        pass


class StreamVocoder(object):

    input_queue: queue.Queue
    output_queue: queue.Queue

    def __init__(self, config) -> None:
        self.vocoder = simple.Vocoder(config)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.chunk_size = 10

        self._warmup()

    def _warmup(self):
        print("warmup...")
        with torch.no_grad():
            _ = self.vocoder.forward(codes=[310] * 10, speaker_id=1)

    def forward(self, x: dict):
        """ forward a single chunk of data """
        st = time.time()
        self.n_tokens = 0
        print(f"Forward: {x}")
        codes = x["message"]
        while len(codes) > 0:  # chunk
            chunk = codes[:self.chunk_size]
            with torch.no_grad():
                self.n_tokens += len(x["message"])
                audio = self.vocoder.forward(codes=chunk, speaker_id=1).detach().cpu().numpy()
                audio = (audio * 32767).astype(np.int16)
            res = {
                "type": "system_audio",
                "data": audio,
                "eos": False,
                "input_timestamp": x['input_timestamp'],
            }
            self.output_queue.put(res)
            codes = codes[self.chunk_size:]
        print(f"Forward: {time.time()-st:.2f}s.")
        tps = self.n_tokens / (time.time()-st)
        print(f"Rate: {int(tps)} tps ({self.n_tokens} tokens).")

    def run(self):
        """ loop to process input queue """
        concat_message = []
        concat_message_timestamp = None
        forward_criterion = ForwardCriterion()
        while True:
            if self.input_queue.empty():
                continue
            else:
                res = self.input_queue.get()
            
            # reset signal
            if res["type"] == "reset":
                print("========== reset all state ==========")
                concat_message = []
                concat_message_timestamp = None
                continue
            
            # system token
            assert res["type"] == "system_token"            
            if res["data"] is not None:  # update system text if not blank (eos is considered blank)
                if res['input_timestamp'] != concat_message_timestamp:  # deal with mind interruption, chunks should have the latest timestamp
                    concat_message_timestamp = res['input_timestamp']
                    concat_message = []
                concat_message.extend(res["data"])

            if forward_criterion.exec(concat_message, res):
                self.forward({
                    "message": concat_message,
                    "input_timestamp": concat_message_timestamp,
                })
                concat_message = []
                concat_message_timestamp = None
            
            # handle system text eos
            eos = res.get("eos", False)
            if eos:
                res = {
                    "type": "system_audio",
                    "data": None,
                    "eos": True,
                    "input_timestamp": res['input_timestamp'],
                }
                self.output_queue.put(res)
                concat_message = []
                concat_message_timestamp = None


class VocoderServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        print("Run Stream Vocoder on new thread!")
        self.model = StreamVocoder(self.config)
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> DefaultProcessor:
        p = DefaultProcessor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
