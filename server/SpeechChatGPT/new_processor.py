import socket
import asyncio
import pickle
import time
import queue
import numpy as np

from .. import Define
from ..utils import Client, AudioLogger
from ..utils import length_prefixing, recv_with_length_prefixing, handle_asyncio_exception


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
    
    async def turn_take_signal(self, check_freq: float=0.1):
        """ return if turn take triggered """
        self.flag = False
        while True:
            if self.last_tick_timestamp is not None and self._get_timestamp() - self.last_tick_timestamp >= 1.0:
                self.flag = True
            if self.flag:
                self.last_tick_timestamp = None
                break
            await asyncio.sleep(check_freq)


class MindHandler(object):
    def __init__(self, client_socket: socket.socket, addr, llm_client: Client) -> None:
        self.queue = queue.Queue()
        self.client_socket = client_socket
        self.addr = addr
        # self._current_task = None
        self.timestamp = None

        self._open = False
        self.llm_client = llm_client
        self.flush_assistant_said = None

        self.running_status = True
    
    async def run(self):
        loop = asyncio.get_event_loop()
        try:
            while self.running_status:  # loops forever
                if self._open:
                    await asyncio.sleep(0.1)  # slower respond to test interruption
                    if not self.queue.empty() and self._open:  # check availability again after sleep
                        res = self.queue.get()
                        await loop.sock_sendall(self.client_socket, length_prefixing(res))
                        if res["data"] is not None:
                            self.flush_assistant_said = await self.llm_client.send_data({
                                "type": "assistant_said",
                                "data": res["data"],
                            })
                        if res.get("eos", False):
                            self.flush_assistant_said = await self.llm_client.send_data({
                                "type": "assistant_said",
                                "data": None,
                                "eos": True
                            })
                            print("[Text] Expression done.")
                            self._open = False
                await asyncio.sleep(0)  # give control to event loop
        except:
            pass
        print("Text mind closed.")

    def express(self):
        self._open = True
        print("[Text] Expression start.")

    def suspend(self):
        if self._open:
            print("[Text] Expression interrupted.")
        else:
            if not self.queue.empty():
                print("[Text] Mind interrupted.")
        self._open = False

    async def put(self, x, timestamp):
        # print(f"Put {x}")
        if self.timestamp is None or timestamp != self.timestamp:
            return
        print(f"[Text] Add to mind: ", x)
        self.queue.put(x)
    
    def set_timestamp(self, x):
        self.timestamp = x
    
    def clear(self):  # blocking
        if not self.queue.empty():
            print("[Text] Clear mind.")
            self.queue = queue.Queue()


class AudioMindHandler(object):
    def __init__(self, client_socket: socket.socket, addr) -> None:
        self.queue = queue.Queue()
        self.client_socket = client_socket
        self.addr = addr
        # self._current_task = None
        self.timestamp = None

        self._open = False
        self.running_status = True

    # def gen_block_data(self, leftover_data: np.ndarray, block_size: int) -> tuple[np.ndarray, bool, np.ndarray]:
    def gen_block_data(self, leftover_data: np.ndarray, block_size: int):
        """ return a fix-sized audio numpy array and eos flag """
        frames_to_read = block_size
        data = []
        length = 0
        eos = False
        while length < frames_to_read:
            if len(leftover_data) >= frames_to_read - length:
                # We have enough leftover data to fulfill the request
                data.append(leftover_data[:frames_to_read - length])
                leftover_data = leftover_data[frames_to_read - length:] if len(leftover_data) > frames_to_read - length else []
                length = frames_to_read
            else:
                if not self.queue.empty():
                    # Get more data from the queue
                    more_data = self.queue.get()
                    # print(more_data)
                    if more_data.get("eos", False):
                        eos = True
                        data.append(np.zeros((frames_to_read - length,)).astype(np.int16))
                        break
                    audio_data = more_data["data"]
                    leftover_data = np.concatenate([leftover_data, audio_data]) if len(leftover_data) else audio_data
                else:
                    data.append(np.zeros((frames_to_read - length,)).astype(np.int16))
                    break
        return data, eos, leftover_data

    async def run(self):
        loop = asyncio.get_event_loop()
        leftover_data = []
        frames_to_read = 4000
        try:
            while self.running_status:  # loops forever
                if self._open:
                    data, eos, leftover_data = self.gen_block_data(leftover_data, block_size=frames_to_read)
                    data = np.concatenate(data)
                    assert len(data) == frames_to_read, "make sure transmit with fix size!"
                    res = {
                        "type": "system_audio",
                        "data": data,
                        "eos": False,
                        "input_timestamp": self.timestamp,
                    }
                    await loop.sock_sendall(self.client_socket, length_prefixing(res))

                    # eos token
                    if eos:
                        print("[Audio] Expression done.")
                        self._open = False
                        res = {
                            "type": "system_audio",
                            "data": None,
                            "eos": True,
                            "input_timestamp": self.timestamp,
                        }
                        await loop.sock_sendall(self.client_socket, length_prefixing(res))
                    await asyncio.sleep(0.23)  # real time synchoronization
                else:
                    leftover_data = []
                await asyncio.sleep(0)  # give control to event loop
        except:
            pass
        print("Audio mind closed.")
    
    def express(self):
        self._open = True
        print("[Audio] Expression start.")

    def suspend(self):
        if self._open:
            print("[Audio] Expression interrupted.")
        else:
            if not self.queue.empty():
                print("[Audio] Mind interrupted.")
        self._open = False

    async def put(self, x, timestamp):
        # print(f"Put {x}")
        if self.timestamp is None or timestamp != self.timestamp:
            return
        print(f"[Audio] Add to mind: ", x)
        self.queue.put(x)
    
    def set_timestamp(self, x):
        self.timestamp = x
    
    def clear(self):  # blocking
        if not self.queue.empty():
            print("[Audio] Clear mind.")
            self.queue = queue.Queue()


class InterLeaving(object):

    def __init__(self, llm_client: Client):
        self.llm_client = llm_client
        self.asr_res_buffer = []
        self.unit_res_buffer = []
        self.TPS= 50 / Define.UNIT_DOWNSAMPLE_RATE # Need to change according to the token per second rate of the speech to unit module. We would need to turn this to be a parameter in config file.
        self.SAMPLE_RATE=16000
        self.check_freq = 0.3

        self.running_status = True

    def add_asr_res(self, x):
        self.asr_res_buffer.append(x)

    def add_unit_res(self, x):
        self.unit_res_buffer.append(x)
    
    async def run(self):
        while self.running_status:  # loops forever
            if len(self.asr_res_buffer) == 0 and len(self.unit_res_buffer) > 0:  # unit only
                kms = []
                input_timestamp = self.unit_res_buffer[-1]["input_timestamp"]
                for x in self.unit_res_buffer:
                    kms += x["data"]
                self.unit_res_buffer = []

                # create interleaving data
                interleaving_data_str = ''.join(kms)
                res = {
                    "type": "user_text",
                    "data": interleaving_data_str,
                    "eos": False,
                    "input_timestamp": input_timestamp
                }
                print(f"[Main Server] Create interleaving data = {interleaving_data_str}")
                await self.llm_client.send_data(res)
            elif len(self.asr_res_buffer) > 0:
                kms, words = [], []
                input_timestamp = self.asr_res_buffer[-1]["input_timestamp"]
                for x in self.asr_res_buffer:
                    for (s, e, word) in x["segment"]:
                        words.append(word)
                self.asr_res_buffer = []
                for x in self.unit_res_buffer:
                    kms += x["data"]
                    input_timestamp = max(input_timestamp, x["input_timestamp"])  # get the latest one
                self.unit_res_buffer = []

                # create interleaving data
                interleaving_data_str = ''.join(kms) + ''.join(words)
                res = {
                    "type": "user_text",
                    "data": interleaving_data_str,
                    "eos": False,
                    "input_timestamp": input_timestamp
                }
                print(f"[Main Server] Create interleaving data = {interleaving_data_str}")
                await self.llm_client.send_data(res)
            else:
                pass
            await asyncio.sleep(self.check_freq)

    def reset(self):
        self.asr_res_buffer = []
        self.unit_res_buffer = []


class Processor(object):
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        if Define.LOG_AUDIO:
            self.audio_logger = AudioLogger("_data/client.wav")
        # Creating text and unit interleaving input data in a streaming way 
        self.unit_accu_data = []
        self.cur_unit_pos = 0
        self.unit_hop_length = Define.UNIT_HOP_LENGTH
        self.condition = asyncio.Condition() # Let ASR and speech to unit module wait for each other to finish
        self.TPS=50/Define.UNIT_DOWNSAMPLE_RATE # Need to change according to the token per second rate of the speech to unit module. We would need to turn this to be a parameter in config file.
        self.SAMPLE_RATE=16000
        # ASR buffer 
        self.asr_buffer = []

        self.running_status = True

    async def emit_close_signal(self) -> None:
        if self.running_status:
            self.running_status = False
            await self.close()

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
        self.system_audio_mind.clear()

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
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print("listen_to_turn_take closed.")
            await self.emit_close_signal()

    async def combine_text_unit(self):
        # Currently no new ASR transcription in the buffer
        if len(self.asr_buffer) == 0:
            unit_s = self.cur_unit_pos*self.TPS
            unit_e = min(self.cur_unit_pos+self.unit_hop_length, len(self.unit_accu_data))*self.TPS
            print(unit_s, unit_e)
            interleaving_data = self.unit_accu_data[round(unit_s):round(unit_e)]
            print(f"Create Unit only Interleaving data for length = {len(interleaving_data)}")
            # self.cur_unit_pos = unit_e
            self.cur_unit_pos = unit_e / self.SAMPLE_RATE / self.TPS
            return ''.join(interleaving_data)
        else:
            # asr_max_time_stamp = self.asr_buffer[-1][1] # Get the time stamp of the end of the last word
            asr_start_time = self.asr_buffer[0][0]
            asr_end_time = self.asr_buffer[-1][1]
            asr_trans_length = asr_end_time-asr_start_time
            required_unit_len = round((self.cur_unit_pos+asr_trans_length)*self.TPS)
            
            while required_unit_len > len(self.unit_accu_data):  # Wait for unit generation
                await self.condition.wait()

            assert required_unit_len <= len(self.unit_accu_data), f"ASR max={required_unit_len},Unit={len(self.unit_accu_data)}" # Assume that unit runs faster than asr 
            asr_s, asr_e = self.cur_unit_pos, self.cur_unit_pos+asr_trans_length
            unit_s, unit_e = round(asr_s*self.TPS), round(asr_e*self.TPS)
            print(f"Unit start={unit_s}, Unit end={unit_e}, ASR start={asr_s}, ASR end={asr_e}")
            print(f"Unit buffer length={len(self.unit_accu_data)}")
            print(f"Create ASR+Unit Interleaving data")
            kms = self.unit_accu_data[max(0,unit_s-1):unit_e]
            segments = [(s-asr_s,w) for s,e,w in self.asr_buffer]
            words = []
            for segment in segments:
                start, word = segment
                words.append((word, int(start * self.TPS)))
            for i, (w, s) in enumerate(words):
                kms.insert(i + s, ' ' + w)

            return ''.join(kms) 

    async def listen_for_asr_server(self):
        loop = asyncio.get_event_loop()
        try:
            stream = self.asr_client.recv_stream()
            async for res in stream:
                print(res)
                if res["type"] == 'hallucinate':
                    res = {"type": "reset"}
                    await asyncio.gather(
                        self.asr_client.send_data(res),
                        self.unit_client.send_data(res),
                    )
                    continue
                self.turn_taker.tick()

                await self.interruption(res)

                res["type"] = "user_text"
                self.interleaving.add_asr_res(res)
                await loop.sock_sendall(self.client_socket, length_prefixing(res))  # send to client instantly
                # self.asr_buffer += res["segment"]
                # if len(self.asr_buffer) >= self.interleave_min_word:
                #     interleaving_data = await self.combine_text_unit()
                #     self.asr_buffer = [] 
                # print(f"[Main Server] Create interleaving data = {interleaving_data}")
                # res["data"] = interleaving_data
                # await self.llm_client.send_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print("ASR server closed.")
            await self.emit_close_signal()

    async def listen_for_unit_server(self):
        loop = asyncio.get_event_loop()
        try:
            stream = self.unit_client.recv_stream()
            async for res in stream:
                # print(res)
                self.turn_taker.tick()

                await self.interruption(res)

                self.interleaving.add_unit_res(res)

                # async with self.condition:
                #     self.unit_accu_data += res["data"]
                #     self.condition.notify_all()
                # interleaving_data = await self.combine_text_unit()
                # print(f"[Main Server] Create interleaving data = {interleaving_data}")
                # res["data"] = interleaving_data
                # await self.llm_client.send_data(res)
        
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print("Unit server closed.")
            # await self.emit_close_signal()

    async def listen_for_llm_server(self):
        try:
            stream = self.llm_client.recv_stream()
            async for res in stream:
                if res["type"] == "system_text":
                    await self.system_text_mind.put(res, timestamp=res["input_timestamp"])
                elif res["type"] == "system_token":
                    await self.tts_client.send_data(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print("LLM server closed.")
            await self.emit_close_signal()

    async def listen_for_tts_server(self):
        try:
            stream = self.tts_client.recv_stream()
            async for res in stream:
                await self.system_audio_mind.put(res, timestamp=res["input_timestamp"])
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            print("Vocoder server closed.")
            await self.emit_close_signal()

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
        res["type"] = "user_text"
        await self.interruption(res)

        if res["data"] == "===":  # special command
            res = {"type": "reset"}
            self.interleaving.reset()
            await asyncio.gather(
                self.tts_client.send_data(res),
                self.llm_client.send_data(res),
                self.asr_client.send_data(res),
                self.unit_client.send_data(res),
            )
            if Define.LOG_AUDIO:
                self.audio_logger.stop()
            return
                    
        await self.llm_client.send_data(res)        
        self.turn_taker.emit_signal()

    async def run(self):
        """ connect to subservers. """
        self.asr_client = Client(self.config["asr"]["host"], self.config["asr"]["port"])
        self.unit_client = Client(self.config["unit"]["host"], self.config["unit"]["port"])
        self.llm_client = Client(self.config["llm"]["host"], self.config["llm"]["port"])
        self.tts_client = Client(self.config["tts"]["host"], self.config["tts"]["port"])
        self.interleaving = InterLeaving(self.llm_client)
        self.turn_taker = TurnTaker()

        try:
            await asyncio.gather(
                self.asr_client.run(),
                self.unit_client.run(),
                self.llm_client.run(),
                self.tts_client.run()
            )
        except:
            print("Connect to subservers failed.")
            raise

        # debug, disable the client by suspending send_data() if you need
        if Define.DEBUG_MODE == "asr":
            self.llm_client.disable()
            self.tts_client.disable()
        elif Define.DEBUG_MODE == "llm":
            self.tts_client.disable()
    
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
                    await self.receive_text(res)
                else:
                    raise NotImplementedError
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
        except Exception as e:
            raise
        finally:
            await self.emit_close_signal()
    
    async def process(self):
        self.system_text_mind = MindHandler(self.client_socket, self.addr, llm_client=self.llm_client)
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
            self.interleaving.run()
        ]
        self.tasks = [asyncio.create_task(coro) for coro in coros]
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            raise
    
    async def close(self):
        print(f"Connection from {self.addr} closed.")
        self.system_text_mind.running_status = False
        self.system_audio_mind.running_status = False
        self.interleaving.running_status = False
        self.client_socket.close()
        # manually cancel input_data and turn_take
        if not self.tasks[0].done():
            self.tasks[0].cancel()
        if not self.tasks[7].done():
            self.tasks[7].cancel()
        if not self.tasks[8].done():
            self.tasks[7].cancel()
        await self.asr_client.close()
        await self.llm_client.close()
        await self.tts_client.close()
        await self.unit_client.close()
        print("Connection gracefully shutdown.")


class MainServerProcessor(object):
    """ This is only a wrapper class. """
    def __init__(self, config):
        self.config = config
    
    async def process(self, client_socket: socket.socket, addr):  # handle one client connection
        loop = asyncio.get_event_loop()
        processor = Processor(self.config, client_socket, addr)
        await processor.run()

        # main process loop
        fut = loop.create_task(processor.process())
        fut.add_done_callback(handle_asyncio_exception)
