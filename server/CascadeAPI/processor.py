import socket
import asyncio
import pickle
import time
import traceback

from .. import Define
from server.base import IProcessor, IServerProcessor
from server.common.MindHandler import TextMindHandler, AudioMindHandler
from server.common.TurnTaker import TurnTaker
from server.utils import Client, AudioLogger
from server.utils import length_prefixing, recv_with_length_prefixing, run_parallel_tasks


class Processor(IProcessor):
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.turn_taker = TurnTaker()

        if Define.LOG_AUDIO:
            self.audio_logger = AudioLogger("_data/client.wav")

    def expression(self) -> None:
        """ Start expressing the **mind**. """
        self.system_text_mind.express()
        self.system_audio_mind.express()

    async def interruption(self, res):
        """ 
        Suspend expression and reset **mind**. Unlike expression, this function is asynchronous since we
        need to wait until assistant said flushed to LLM's chat history.
        """
        self.system_text_mind.suspend()
        self.system_text_mind.set_timestamp(res["input_timestamp"])
        self.system_text_mind.clear()

        # flush assistant said (e.g. update llm chat history)
        if self.system_text_mind.flush_assistant_said and not self.system_text_mind.flush_assistant_said.done():
            await self.system_text_mind.flush_assistant_said

        self.system_audio_mind.suspend()
        self.system_audio_mind.set_timestamp(res["input_timestamp"])

    async def _take_turn(self, role: str):
        loop = asyncio.get_event_loop()
        await loop.sock_sendall(self.client_socket, length_prefixing({
            "type": "command",
            "data": f"{role}_turn"
        }))
    
    async def listen_to_turn_take(self):
        loop = asyncio.get_event_loop()
        try:
            while True:
                await self.turn_taker.turn_take_signal()
                print("turn take!")
                eos = {
                    "type": "user_text",
                    "data": None,
                    "eos": True,
                    "input_timestamp": time.time()
                }
                await self.llm_client.send_data(eos)
                await loop.sock_sendall(self.client_socket, length_prefixing(eos))
                self.expression()
                await self._take_turn("system")
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def listen_for_asr_server(self):
        loop = asyncio.get_event_loop()
        try:
            stream = self.asr_client.recv_stream()
            async for res in stream:
                if res["type"] == 'hallucinate':
                    res = {"type": "reset"}
                    await self.asr_client.send_data(res)
                    continue
                await self.turn_taker.tick()

                await self.interruption(res)
                await self._take_turn("user")

                res["type"] = "user_text"
                print(f"user_text:", res["data"])
                await loop.sock_sendall(self.client_socket, length_prefixing(res))  # send to client instantly
                await self.llm_client.send_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def listen_for_llm_server(self):
        try:
            stream = self.llm_client.recv_stream()
            async for res in stream:
                if res["input_timestamp"] != self.system_text_mind.timestamp:
                    continue
                await self.system_text_mind.put(res, timestamp=res["input_timestamp"])
                await self.tts_client.send_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def listen_for_tts_server(self):
        try:
            stream = self.tts_client.recv_stream()
            async for res in stream:
                if res["input_timestamp"] != self.system_audio_mind.timestamp:
                    continue
                await self.system_audio_mind.put(res, timestamp=res["input_timestamp"])
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def receive_audio(self, res):
        # print(f"audio!: {res['data'].shape}")
        await self.asr_client.send_data(res)
        if Define.LOG_AUDIO:
            self.audio_logger.write(res)

    async def receive_text(self, res):
        # print(f"text!: {res['data']}")
        # res["type"] = "user_text"
        await self.interruption(res)

        if res["data"] == "===":  # special command
            res = {"type": "reset"}
            await self.turn_taker.interrupt()
            await asyncio.gather(
                self.tts_client.send_data(res),
                self.llm_client.send_data(res),
                self.asr_client.send_data(res),
            )
            if Define.LOG_AUDIO:
                self.audio_logger.stop()
            await self._take_turn("user")
            return
        
        await self.llm_client.send_data(res)        
        self.turn_taker.emit_signal()

    async def _connect_subservers(self):
        """ connect to subservers. """
        self.asr_client = Client(self.config["asr"]["host"], self.config["asr"]["port"])
        self.llm_client = Client(self.config["llm"]["host"], self.config["llm"]["port"])
        self.tts_client = Client(self.config["tts"]["host"], self.config["tts"]["port"])

        try:
            coros = [
                self.asr_client.run(),
                self.llm_client.run(),
                self.tts_client.run()
            ]
            tasks = [asyncio.create_task(coro) for coro in coros]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # check connection is successful or not
            for result in results:
                if isinstance(result, BaseException):
                    raise result

            # debug, disable the client by suspending send_data() if you need
            if Define.DEBUG_MODE == "asr":
                self.llm_client.disable()
                self.tts_client.disable()
            elif Define.DEBUG_MODE == "llm":
                self.tts_client.disable()
        except Exception as e:
            print("Connect to subservers failed.")
            await self.close()
            raise e
    
    async def input_data(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                # print(res)
                if not res:
                    break
                res = pickle.loads(res)
                if res["type"] == "audio":  # should return user_text(transcription), system_text, system_audio
                    await self.receive_audio(res)
                elif res["type"] == "text":  # should return system_text, system_audio
                    res["type"] = "user_text"
                    await self.receive_text(res)
                elif res["type"] == "user_text":  # should return system_text, system_audio
                    await self.receive_text(res)
                elif res["type"] == "system_prompt":  # should return system_text, system_audio
                    await self.receive_text(res)
                else:
                    raise NotImplementedError
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise
        print(f"Connection from {self.addr} closed.")
    
    async def exec(self):
        self.system_text_mind = TextMindHandler(self.client_socket, self.addr, llm_client=self.llm_client)
        self.system_audio_mind = AudioMindHandler(self.client_socket, self.addr)

        coros = [
            self.input_data(),
            self.system_text_mind.run(),
            self.system_audio_mind.run(),
            self.listen_for_asr_server(),
            self.listen_for_llm_server(),
            self.listen_for_tts_server(),
            self.listen_to_turn_take(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        await self.close()

    async def close(self):
        self.client_socket.close()
        await self.asr_client.close()
        await self.llm_client.close()
        await self.tts_client.close()


class MainServerProcessor(IServerProcessor):
    def __init__(self, config):
        self.config = config

    async def process(self, client_socket: socket.socket, addr):  # handle one client connection
        try:
            processor = Processor(self.config, client_socket, addr)
            await processor._connect_subservers()
            await processor.exec()
        except Exception as e:
            traceback.print_exc()
        finally:
            del processor
            print(f"Connection from {addr} gracefully shutdown.")
