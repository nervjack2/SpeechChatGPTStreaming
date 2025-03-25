import os
import socket
import asyncio
import time
import pickle
import traceback

from openai import AsyncOpenAI

from server.base import IProcessor
from server.common.template import DefaultServerProcessor
from server.utils import recv_with_length_prefixing, length_prefixing, run_parallel_tasks
from ..SpeechLM.utils import LMState


class Processor(IProcessor):

    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.model_name = None
        self.model_client = None
        self.lm_state = LMState()
        self.start_infer_timestamp = None
        self.user_end_timestamp = None
        self.fut = None
        self._api_cnt = 0

    def connect_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.lm_state.update_history("system", self.get_system_prompt())

    async def reset(self) -> None:
        await self._interrupt_infer_loop()
        del self.model_client
        print("========== reset all state ==========")
        self.model_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.start_infer_timestamp = None
        self.user_end_timestamp = None
        self._api_cnt = 0
        self.lm_state.reset()
        self.lm_state.update_history("system", self.get_system_prompt())
    
    def get_system_prompt(self, prompt: str=None) -> str:
        if prompt is None:
            return """You are an intelligent conversational AI assistant communicating with the user in real-time voice mode. Key points to keep in mind:
- The user may interrupt you mid-response. Be prepared to pause and adjust your reply accordingly.
- The conversation may be fragmented, try to maintain context awareness.
- If the user doesn't say anything, you can continue to talk or make an ending.
- The user may not use proper punctuation, wait for user to finished their sentence.
- You can ask questions to confirm the user's needs.
"""
        else:
            return prompt
        
    # inference functions
    async def _infer_loop(self):
        """ main inference loop """
        loop = asyncio.get_event_loop()
        print(f"Send to OpenAI: {self.lm_state.chat_history[1:]}")
        self._api_cnt += 1
        stream = await self.model_client.chat.completions.create(
            model=self.model_name,
            messages=self.lm_state.chat_history,
            stream=True  # Enable streaming mode
        )
        input_timestamp = self.user_end_timestamp
        async for chunk in stream:
            if chunk:
                # print(chunk.model_dump_json(indent=4))
                content = chunk.choices[0].delta.content
                res = {
                    "type": "system_text",
                    "data": content,
                    "eos": (content is None),
                    "input_timestamp": input_timestamp,
                }
                print(f"Recv from OpenAI: {res}")
                await loop.sock_sendall(self.client_socket, length_prefixing(res))
                if res["eos"]:  # end of stream
                    break

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
            return
        self.lm_state.update_history("assistant", res["data"])

    async def _handle_eos(self, res):
        print("turn take!")
        if self.fut is None and self.user_end_timestamp is not None:  # make sure user did input something
            await self._start_infer()
    
    async def _handle_system_prompt(self, res):
        # assert self.user_end_timestamp is None, "System prompt should be set at the very beginning."
        system_prompt = self.get_system_prompt(res["data"])
        self.lm_state.update_history("system", system_prompt)

    async def _handle_user_text(self, res):
        await self._interrupt_infer_loop()
        self.user_end_timestamp = res["input_timestamp"]

        # set state
        self.lm_state.update_history("user", res["data"])
        # await self._update_lm_cache()
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
        assert self.model_name is not None, "Please connect the model by calling connect_model(model) first"
        coros = [
            self.input_data(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        self.client_socket.close()


class GPTServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        # self.model = "gpt-4"
        self.model = "gpt-4o-mini"
        print(f"Using {self.model} from OpenAI!")

    def _create_processor(self, client_socket: socket.socket, addr) -> Processor:
        p = Processor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
