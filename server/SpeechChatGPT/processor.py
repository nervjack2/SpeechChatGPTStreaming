import socket
import asyncio
import pickle
import time
import queue
import numpy as np
import traceback

from .. import Define
from server.base import IProcessor, IServerProcessor
from server.common.MindHandler import TextMindHandler, AudioMindHandler
# from server.common.TurnTaker import TurnTaker
from server.utils import Client, AudioLogger
from server.utils import length_prefixing, recv_with_length_prefixing, run_parallel_tasks


class TurnTaker(object):
    def __init__(self) -> None:
        self.time_base = time.time()
        self.last_tick_timestamp = None
        self.flag = False

    def _get_timestamp(self) -> float:
        return time.time() - self.time_base
    
    def tick(self):
        self.last_tick_timestamp = self._get_timestamp()

    def emit_signal(self):
        """ trigger turn take """
        self.flag = True
    
    async def turn_take_signal(self, check_freq: float=0.02):
        """ return if turn take triggered """
        self.flag = False
        while True:
            if self.last_tick_timestamp is not None and self._get_timestamp() - self.last_tick_timestamp >= 1.0:
                self.flag = True
            if self.flag:
                self.last_tick_timestamp = None
                break
            await asyncio.sleep(check_freq)


class Processor(IProcessor):
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.turn_taker = TurnTaker()

        if Define.LOG_AUDIO:
            self.audio_logger = AudioLogger("_data/client.wav")
        # Creating text and unit interleaving input data in a streaming way 
        self.unit_accu_data = []
        self.condition = asyncio.Condition() # Let ASR and speech to unit module wait for each other to finish
        self.TPS=50/Define.UNIT_DOWNSAMPLE_RATE # Need to change according to the token per second rate of the speech to unit module. We would need to turn this to be a parameter in config file.
        self.SAMPLE_RATE=16000
        # ASR buffer 
        self.asr_buffer = []
        self.interleave_min_word = Define.INTERLEAVE_MIN_WORD
        self.unit_last_end = 0
        self.max_remain_unit_sec = Define.MAX_REMAIN_UNIT_SEC

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
                async with self.condition:
                    print("Reset ASR and Unit server!")
                    res = {"type": "reset"}
                    await asyncio.gather(
                        self.asr_client.send_data(res),
                        self.unit_client.send_data(res),
                    )
                    self.unit_accu_data = []
                    self.asr_buffer = []
                    self.unit_last_end = 0
                await self.llm_client.send_data(eos)
                await loop.sock_sendall(self.client_socket, length_prefixing(eos))
                self.expression()
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def combine_text_unit(self):
        async with self.condition:
            asr_max_time_stamp = self.asr_buffer[-1][1] # Get the time stamp of the end of the last word
            required_unit_len = round(asr_max_time_stamp*self.TPS)
            
            while required_unit_len > len(self.unit_accu_data):  # Wait for unit generation
                await self.condition.wait()

            assert required_unit_len <= len(self.unit_accu_data), f"ASR max={required_unit_len},Unit={len(self.unit_accu_data)}" # Assume that unit runs faster than asr 
            if not len(self.asr_buffer):
                return ''
            asr_s, asr_e = self.asr_buffer[0][0], self.asr_buffer[-1][1]
            unit_s, unit_e = round(asr_s*self.TPS), round(asr_e*self.TPS)
            print(f"Unit start={unit_s}, Unit end={unit_e}, ASR start={asr_s}, ASR end={asr_e}")
            print(f"Unit buffer length={len(self.unit_accu_data)}")
            kms = self.unit_accu_data[max(0,unit_s-1):unit_e]
            segments = [(s-asr_s,w) for s,e,w in self.asr_buffer]
            words = []
            for segment in segments:
                start, word = segment
                words.append((word, int(start * self.TPS)))
            for i, (w, s) in enumerate(words):
                kms.insert(i + s, ' ' + w)
            
            if (max(0,unit_s-1)-self.unit_last_end) > (self.max_remain_unit_sec*self.TPS):
                self.unit_last_end = round(max(0,unit_s-1)-(self.max_remain_unit_sec*self.TPS))

            remain_kms = self.unit_accu_data[self.unit_last_end:max(0,unit_s-1)]
            self.unit_last_end = unit_e
            kms = remain_kms + kms

            return ''.join(kms) 

    async def listen_for_asr_server(self):
        loop = asyncio.get_event_loop()
        try:
            stream = self.asr_client.recv_stream()
            async for res in stream:
                print("ASR: ", res)
                if res["type"] == 'hallucinate':
                    async with self.condition:
                        res = {"type": "reset"}
                        self.unit_accu_data = []
                        self.asr_buffer = []
                        self.unit_last_end = 0
                        await asyncio.gather(
                            self.asr_client.send_data(res),
                            self.unit_client.send_data(res),
                        )
                    continue
                
                # interleaving
                # print(f"user_text:", res["data"], res["accu_len"], res["segment"])
                self.asr_buffer += res["segment"]
                if len(self.asr_buffer) >= self.interleave_min_word:
                    interleaving_data = await self.combine_text_unit()
                    self.asr_buffer = [] 
                    if not len(interleaving_data):
                        continue
                print(f"[Main Server] Create interleaving data = {interleaving_data}")

                # data confirmed
                self.turn_taker.tick()
                await self.interruption(res)
                await loop.sock_sendall(self.client_socket, length_prefixing(res))  # send to client
                res["data"] = interleaving_data
                await self.llm_client.send_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    async def listen_for_unit_server(self):
        try:
            stream = self.unit_client.recv_stream()
            async for res in stream:
                # print(res)
                # self.turn_taker.tick()

                # await self.interruption(res)

                # print(f"user_unit:", res["data"], res["accu_len"])
                # Informing coroutines after adding new datas
                async with self.condition:
                    self.unit_accu_data += res["data"]
                    self.condition.notify_all()
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
                if res["type"] == "system_text":
                    await self.system_text_mind.put(res, timestamp=res["input_timestamp"])
                elif res["type"] == "system_token":
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
        await asyncio.gather(
            self.asr_client.send_data(res),
            self.unit_client.send_data(res),
        )
        if Define.LOG_AUDIO:
            self.audio_logger.write(res)

    async def receive_text(self, res):
        # print(f"text!: {res['data']}")
        # res["type"] = "user_text"
        await self.interruption(res)
        await self.llm_client.send_data(res)        
        self.turn_taker.emit_signal()

    async def receive_reset(self, res):
        await self.interruption(res)
        async with self.condition:
            res = {"type": "reset"}
            self.unit_accu_data = []
            self.asr_buffer = []
            self.unit_last_end = 0
            await asyncio.gather(
                self.tts_client.send_data(res),
                self.llm_client.send_data(res),
                self.asr_client.send_data(res),
                self.unit_client.send_data(res),
            )
        if Define.LOG_AUDIO:
            self.audio_logger.stop()
        print("========== reset ==========")
        return
    
    async def _connect_subservers(self):
        """ connect to subservers. """
        self.asr_client = Client(self.config["asr"]["host"], self.config["asr"]["port"])
        self.unit_client = Client(self.config["unit"]["host"], self.config["unit"]["port"])
        self.llm_client = Client(self.config["llm"]["host"], self.config["llm"]["port"])
        self.tts_client = Client(self.config["tts"]["host"], self.config["tts"]["port"])

        try:
            coros = [
                self.asr_client.run(),
                self.unit_client.run(),
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
                elif res["type"] == "user_text":  # should return system_text, system_audio
                    if res["data"] == "===":
                        await self.receive_reset(res)
                    else:
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
            self.listen_for_unit_server(),
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
        await self.unit_client.close()


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
