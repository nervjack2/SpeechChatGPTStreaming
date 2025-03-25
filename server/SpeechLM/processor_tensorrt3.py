import socket
import asyncio
import numpy as np
import torch
import time
from threading import Thread
import queue
import pickle
import uuid
import traceback
import copy
from transformers import AutoTokenizer

from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.hlapi import SamplingParams

from server.base import IProcessor
from server.common.template import DefaultServerProcessor
from server.common.ModelClient import IServing, AsyncModelClient
from server.utils import recv_with_length_prefixing, length_prefixing, run_parallel_tasks
from .utils import LMState


class SpokenLlama(IServing):
    input_queue: queue.Queue
    output_queues: dict[uuid.UUID, queue.Queue]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_queue = queue.Queue()
        self.output_queues = {}

        self._build_model(config)
        self._warmup()

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
    
    def get_system_prompt(self, prompt: str=None) -> str:
        if prompt is None:
            assert "model_name" in self.config
            if self.config["model_name"] in ["taipei1/speechllama3.1-v0"]:
                return "You are a chatbot, only response precisely. Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
            elif self.config["model_name"] in ["taipei1/speechllama3.1-sft-with-back", "taipei1/spml-omni-instruct-step19698"]:
                return "你是一個能和人類溝通與交流的聊天機器人。今天你正在和使用者進行一段有趣的對談 Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
            else:
                raise NotImplementedError
        else:
            return prompt + " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
        
    def _build_model(self, config: dict) -> None:
        # Initialize the model and tokenizer
        model_name = config["model_name"]
        model_path = config["model_path"]
        if model_path == "":
            model_path = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

        self.llm = GenerationExecutor.create(config["tensorrt_engine_dir"])
        print("Model loaded")

    def _warmup(self):
        print("warmup...")
        input_str = "What should you say when someone gives you a gift? You should say:"
        _ = self.llm.generate(
            self.tokenizer.encode(input_str),
            sampling_params=SamplingParams(max_tokens=10)
        )

    # output parsing
    def _is_speech_unit(self, token_id) -> bool:
        return 128256 <= token_id < 130256
    
    def _post_process(self, result_str: str):
        kms = []
        words = []
        interleave, token_mask = [], []
        while result_str:
            if result_str[:2] == '<|':
                idx = result_str.find('|>')
                kms.append(result_str[2:idx])
                interleave.append(result_str[2:idx])
                token_mask.append(True)
                result_str = result_str[idx+2:]
            else:
                idx = result_str.find('<|')
                if idx == -1:
                    idx = len(result_str)
                words.append(result_str[:idx])
                interleave.append(result_str[:idx])
                token_mask.append(False)
                result_str = result_str[idx:]
        return kms, words, interleave, token_mask
    
    def _generate_n_steps(self, input_ids: list[int], gen_len: int=10):
        output = self.llm.generate(
            input_ids,
            sampling_params=SamplingParams(max_tokens=gen_len, top_k=30)
        )
        return output.outputs[0].token_ids
    
    # handler functions
    def _handle_decode(self, req):
        headers, body = req["headers"], req["request_body"]
        generated_ids = self._generate_n_steps(
            input_ids=body["input_ids"],
            gen_len=body["max_gen_len"]
        )
        self.output_queues[headers["uid"]].put({
            "headers": headers,
            "data": {
                "generated_ids": generated_ids,
            }
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

            if body["type"] == "decode":
                self._handle_decode(req)
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
        self.lm_state = LMState()
        self.start_infer_timestamp = None
        self.user_end_timestamp = None
        self.fut = None

        self.n_gen = 32
    
    def connect_model(self, model: SpokenLlama) -> None:
        self.model = model
        self.model_client = AsyncModelClient(model)
        self.lm_state.update_history("system", self.model.get_system_prompt())

    async def reset(self) -> None:
        await self._interrupt_infer_loop()
        await self.model_client.close()
        del self.model_client
        print("========== reset all state ==========")
        self.model_client = AsyncModelClient(self.model)
        self.start_infer_timestamp = None
        self.user_end_timestamp = None
        self.lm_state.reset()
        self.lm_state.update_history("system", self.model.get_system_prompt())
    
    # inference functions
    def _parse_generated_ids(self, generate_ids: list[int]):
        ids = []
        eos = False
        ed = 0
        for i, id in enumerate(generate_ids):
            eos = (id == self.model.tokenizer.eos_token_id)
            if self.model._is_speech_unit(id) or eos:
                ids += generate_ids[ed:i+1]
                ed = i + 1
                if eos:
                    break
        remain_ids = generate_ids[ed:]

        # decode
        token = (
            self.model.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
        )
        print("Decoded: ", token)
        kms, words, interleave, token_mask = self.model._post_process(token)
        return remain_ids, kms, words, interleave, token_mask, eos

    async def _infer_loop(self):
        """ main inference loop """
        loop = asyncio.get_event_loop()
        eos_flag = False
        while not eos_flag:
            with_generation_prompt = self.model.tokenizer.apply_chat_template(self.lm_state.chat_history, tokenize=False, add_generation_prompt=True)
            input_ids = self.model.tokenizer.encode(with_generation_prompt, add_special_tokens=False)
            input_ids = input_ids + self.lm_state.full_generated_ids

            model_res = await self.model_client.send_request(req={
                "type": "decode",
                "input_ids": input_ids,
                "max_gen_len": self.n_gen,
            })
            
            # parse
            remain_ids = self.lm_state.generated_ids + model_res["generated_ids"]
            remain_ids, kms, words, interleave, token_mask, eos = self._parse_generated_ids(remain_ids)
            
            # update LMState
            self.lm_state.generated_ids = remain_ids
            self.lm_state.full_generated_ids += model_res["generated_ids"]
            n_tokens = len(self.lm_state.full_generated_ids) - len(remain_ids)
            t_passed = time.perf_counter() - self.start_infer_timestamp
            tps = n_tokens / t_passed
            print(f"Total infer time: {t_passed:.2f}s. Rate: {int(tps)} tps ({n_tokens} tokens).")

            # send back to client
            # print("Processed: ", kms, words)
            input_timestamp = self.user_end_timestamp
            if interleave:
                res_interleave = {
                    "type": "system_interleave",
                    "data": interleave,
                    "token_mask": token_mask,
                    "eos": False,
                    "input_timestamp": input_timestamp,
                }
                # print(f"Recv from Spoken Llama: {res_interleave}")
                await loop.sock_sendall(self.client_socket, length_prefixing(res_interleave))
            
            # eos
            if len(self.lm_state.full_generated_ids) > 1000:  # force end
                eos = True
            if eos:
                self.lm_state.generated_ids.clear()
                self.lm_state.full_generated_ids.clear()
                self.user_end_timestamp = None
                res_interleave = {
                    "type": "system_interleave",
                    "data": None,
                    "eos": True,
                    "input_timestamp": input_timestamp,
                }
                await loop.sock_sendall(self.client_socket, length_prefixing(res_interleave))
                # print(f"Recv from Spoken Llama: <eos>")
                eos_flag = True

    async def infer_loop(self):
        try:  # handle exceptions internally
            await self._infer_loop()
            print("Inference completed.")
        except asyncio.CancelledError:
            print("Inference interrupted.")
        except Exception as e:
            traceback.print_exc()

    async def _interrupt_infer_loop(self):
        if self.fut is None or self.fut.done():
            pass
        else:
            self.fut.cancel()
            await self.fut
        self.user_end_timestamp = None
        self.fut = None

    async def _update_lm_cache(self):
        pass

    async def _start_infer(self):
        print("Start infer!")
        self.start_infer_timestamp = time.perf_counter()
        
        # prepare to start infer
        self.lm_state.generated_ids.clear()
        self.lm_state.full_generated_ids.clear()

        # start infer loop
        self.fut = asyncio.create_task(self.infer_loop())

    # handler functions
    async def _handle_assistant_said(self, res):
        if res["data"] is None:
            # print("========== Chat history ==========")
            # for c in self.lm_state.chat_history:
            #     print(c)
            # print("==================================")
            return
        self.lm_state.update_history("Machine", res["data"])

    async def _handle_eos(self, res):
        print("turn take!")
        if self.fut is None and self.user_end_timestamp is not None:  # make sure user did input something
            await self._start_infer()
    
    async def _handle_system_prompt(self, res):
        # assert self.user_end_timestamp is None, "System prompt should be set at the very beginning."
        system_prompt = self.model.get_system_prompt(res["data"])
        self.lm_state.update_history("system", system_prompt)

    async def _handle_user_text(self, res):
        await self._interrupt_infer_loop()
        self.user_end_timestamp = res["input_timestamp"]

        # set state
        self.lm_state.update_history("User", res["data"])
        # await self._update_lm_cache()  # still finding solutions to apply chat history cache on tensorRT
        await self._start_infer()  # always jumped the gun

    async def input_data(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                if not res:
                    break
                res = pickle.loads(res)

                # print(f"Put {res}")
                if res["type"] == "assistant_said":  # update assistant history
                    await self._handle_assistant_said(res)
                elif res["type"] == "reset":  # reset signal
                    await self.reset()
                elif res.get("eos", False):  # turn take when user stops a while
                    await self._handle_eos(res)
                elif res["type"] == "user_text":
                    await self._handle_user_text(res)
                elif res["type"] == "system_prompt":
                    await self._handle_system_prompt(res)
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
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        self.client_socket.close()
        await self.model_client.close()


class SpeechLMServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)
    
    def _setup_model(self):
        print("Run Stream Spoken Llama on new thread!")
        self.model = SpokenLlama(self.config["model"])  # all processors will share one model
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> Processor:
        p = Processor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
