import os
import socket
import asyncio
import time
import pickle
import queue
import traceback

from server.base import IProcessor
from server.common.template import DefaultServerProcessor
from server.utils import recv_with_length_prefixing, length_prefixing, run_parallel_tasks
from .openai import AsyncModelClient


class Processor(IProcessor):
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.model_name = None
        self.model_client = None

        self.concat_data = ""
        self.concat_data_timestamp = None
        self.eos = False
        
        self.req_cnt = 0
        self.res_cnt = 0
        self.request_queue = asyncio.Queue()
        self.pending_requests = {}
        self.is_pending = asyncio.Event()
        self.is_all_done = asyncio.Event()
        self.is_all_done.set()

    def connect_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_client = AsyncModelClient(
            model_name=self.model_name,
            language_code="en-US",
            sr=16000
        )

    async def reset(self) -> None:
        await self._interrupt_infer()
        del self.model_client
        print("========== reset all state ==========")
        self.model_client = AsyncModelClient(
            model_name=self.model_name,
            language_code="en-US",
            sr=16000
        )
        self.concat_data = ""
        self.concat_data_timestamp = None
        self.eos = False
    
    # inference functions
    def _is_infer(self, concat_message, res) -> bool:
        if concat_message == "":
            return False
        if res.get("eos", False):
            return True
        if self.config["segmentation"] == "fixlength":
            is_breaking = len(concat_message) >= 10
        elif self.config["segmentation"] == 'comma':
            is_breaking = res["data"] in [".", "?", "!", "。", "？", "！", "，"]
        elif self.config["segmentation"] == 'punctuation':
            is_breaking = res["data"] in [".", "?", "!", "。", "？", "！"]
        else:
            raise
        return is_breaking
    
    async def _interrupt_infer(self):
        await self.model_client.close()
        await self.is_all_done.wait()
    
    async def _request_loop(self):
        async def wrap_request(req):
            if req.get("eos", False):
                res = {
                    "type": "system_audio",
                    "data": None,
                    "eos": True,
                    "input_timestamp": req['input_timestamp'],
                }
            else:
                audio = await self.model_client.text_to_wav(req["message"])
                res = {
                    "type": "system_audio",
                    "data": audio,
                    "eos": False,
                    "input_timestamp": req['input_timestamp'],
                }
            return res

        while True:
            req = await self.request_queue.get()
            print(f"Send request ({self.req_cnt}):", req)
            self.pending_requests[self.req_cnt] = asyncio.create_task(wrap_request(req))
            self.req_cnt += 1  # ensure ordering
            if self.req_cnt > self.res_cnt:
                # print("pend!")
                self.is_pending.set()
                self.is_all_done.clear()

    async def _response_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            await self.is_pending.wait()  # check if there exists any request under pending
            try:
                res = await self.pending_requests[self.res_cnt]
                print(f"Receive request ({self.res_cnt})")
                await loop.sock_sendall(self.client_socket, length_prefixing(res))
            except:  # request is dead due to reset()
                print(f"Request ({self.res_cnt}) dead due to reset.")
            self.res_cnt += 1  # ensure ordering
            if self.res_cnt == self.req_cnt:
                self.is_pending.clear()
                self.is_all_done.set()
            
    # handler functions    
    async def _handle_system_text(self, res):
        # deal with mind interruption, chunks should have the latest timestamp
        if res['input_timestamp'] != self.concat_data_timestamp:
            self.concat_data_timestamp = res['input_timestamp']
            self.concat_data = ""

        if res["data"] is not None:  # update system text if not blank (eos is considered blank)
            self.eos = False
            self.concat_data += res["data"]

        # handle system text eos
        if res.get("eos", False):
            self.eos = True

        # determine infer
        if self._is_infer(self.concat_data, res):
            await self.request_queue.put({
                "message": self.concat_data,
                "input_timestamp": self.concat_data_timestamp,
            })
            self.concat_data = ""
            self.concat_data_timestamp = None
        if self.eos:
            await self.request_queue.put({
                "eos": True,
                "input_timestamp": res['input_timestamp'],
            })
    
    async def input_data(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                if not res:
                    break
                res = pickle.loads(res)

                # print(f"Put {res}")
                if res["type"] == "system_text":
                    await self._handle_system_text(res)
                elif res["type"] == "system_interleave":
                    if not res.get("eos", False):
                        texts = [x for (x, is_token) in zip(res["data"], res["token_mask"]) if not is_token]
                        for text in texts:  # fine-grained chunking
                            res["data"] = text
                            await self._handle_system_text(res)
                    else:
                        await self._handle_system_text(res)
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
        assert self.model_name is not None, "Please connect the model by calling connect_model(model) first"
        coros = [
            self.input_data(),
            self._request_loop(),
            self._response_loop(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        self.client_socket.close()


class OpenAITTSServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        self.model = "tts-1"
        print(f"Using {self.model} from OpenAI!")

    def _create_processor(self, client_socket: socket.socket, addr) -> Processor:
        p = Processor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
