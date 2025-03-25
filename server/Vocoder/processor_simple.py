import socket
import asyncio
import numpy as np
import torch
import time
from threading import Thread
import queue
import pickle
import uuid

from server.base import IProcessor
from server.common.template import DefaultServerProcessor
from server.common.ModelClient import IServing, AsyncModelClient
from server.utils import recv_with_length_prefixing, length_prefixing, run_parallel_tasks
from .chunker import DynamicChunker
from . import simple


class Vocoder(IServing):

    input_queue: queue.Queue
    output_queues: dict[uuid.UUID, queue.Queue]

    def __init__(self, config) -> None:
        self.vocoder = simple.Vocoder(config)
        self.input_queue = queue.Queue()
        self.output_queues = {}

        self._warmup()

        self.samples_per_unit = 320

    def setup_connection(self, uid):
        self.output_queues[uid] = queue.Queue()
        return self.output_queues[uid], self.input_queue
    
    def close_connection(self, uid):
        self.input_queue.put({
            "headers": {
                "uid": uid,
            },
            "request_body": {"type": "close"},
        })
    
    def _warmup(self):
        print("warmup...")
        with torch.no_grad():
            _ = self.vocoder.forward(codes=[310] * 10, speaker_id=1)

    def forward(self, x: dict):
        """ forward a single chunk of data """
        st = time.time()
        self.n_tokens = 0
        # print(f"Forward: {x}")
        codes = x["message"]
        with torch.no_grad():
            self.n_tokens += len(x["message"])
            audio = self.vocoder.forward(codes=codes, speaker_id=1).detach().cpu().numpy()
            audio = (audio * 32767).astype(np.int16)

        tps = self.n_tokens / (time.time()-st)
        print(f"Forward: {time.time()-st:.2f}s. Rate: {int(tps)} tps ({self.n_tokens} tokens). Shape: {len(audio)}.")

        return {"audio": audio}

    def _handle_synthesize(self, req):
        headers, body = req["headers"], req["request_body"]
        self.output_queues[headers["uid"]].put({
            "headers": headers,
            "data": self.forward({"message": body["data"]}),
        })
    
    def _handle_close(self, req):
        uid = req["headers"]["uid"]
        del self.output_queues[uid]

    def run(self):
        """ loop to process input queue """
        while True:
            req = self.input_queue.get()
            headers, body = req["headers"], req["request_body"]
            if headers["uid"] not in self.output_queues:  # connection already closed
                continue

            if body["type"] == "synthesize":
                self._handle_synthesize(req)
            elif body["type"] == "close":
                self._handle_close(req)
            else:
                raise NotImplementedError


class Processor(IProcessor):
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.model = None
        self.model_client = None
        self.chunker = DynamicChunker()

    def connect_model(self, model) -> None:
        self.model = model
        self.model_client = AsyncModelClient(model)

    async def reset(self) -> None:
        self.chunker.reset()
        await self.model_client.close()
        del self.model_client
        print("========== reset all state ==========")
        self.model_client = AsyncModelClient(self.model)
    
    # handler functions    
    def _handle_system_token(self, res):
        self.chunker.input_data(res)
    
    async def listen_to_chunker(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                chunk_res = self.chunker.output_queue.get_nowait()
                # eos
                if chunk_res.get("eos", False):
                    res = {
                        "type": "system_audio",
                        "data": None,
                        "eos": True,
                        "input_timestamp": chunk_res["input_timestamp"],
                    }
                    await loop.sock_sendall(self.client_socket, length_prefixing(res))
                    continue

                # sythesize
                try:
                    audio_res = await self.model_client.send_request(req={
                        "type": "synthesize",
                        "data": chunk_res["data"],
                    })
                except asyncio.CancelledError:
                    continue
                res = {
                    "type": "system_audio",
                    "data": audio_res["audio"],
                    "eos": False,
                    "input_timestamp": chunk_res["input_timestamp"],
                }
                await loop.sock_sendall(self.client_socket, length_prefixing(res))
            except queue.Empty:
                await asyncio.sleep(0.01)  # return control to event loop

    async def input_data(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                if not res:
                    break
                res = pickle.loads(res)

                # print(f"Put {res}")
                if res["type"] == "system_token":
                    self._handle_system_token(res)
                elif res["type"] == "system_interleave":
                    if not res.get("eos", False):
                        tokens = [x for (x, is_token) in zip(res["data"], res["token_mask"]) if is_token]
                        res["data"] = tokens
                    self._handle_system_token(res)
                elif res["type"] == "reset":  # reset signal
                    await self.reset()
                else:
                    raise NotImplementedError
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise
        print(f"Connection from {self.addr} closed.")

    async def exec(self):
        assert self.model is not None, "Please connect the model by calling connect_model(model) first"
        coros = [
            self.input_data(),
            self.chunker.run(),
            self.listen_to_chunker(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        self.client_socket.close()
        await self.model_client.close()


class VocoderServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        print("Run Stream Vocoder on new thread!")
        self.model = Vocoder(self.config)
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> Processor:
        p = Processor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
