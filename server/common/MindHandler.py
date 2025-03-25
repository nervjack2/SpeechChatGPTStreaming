import socket
import asyncio
import queue
import threading
import numpy as np
import time

from server.utils import Client, length_prefixing


class TextMindHandler(object):
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
        self.output_queue = queue.Queue()
        self.client_socket = client_socket
        self.addr = addr
        self.timestamp = None

        self.chunker = Chunker(channel=self.queue)
        self.mouth = Mouth(
            callback=self.speak_callback,
            samplerate=16000,
            freq=0.05,
            id="Audio"
        )

    def speak_callback(self, frame_count: int):
        data, eos = self.chunker.gen_block_data(frame_count)
        assert len(data) == frame_count
        res = {
            "type": "system_audio",
            "data": data,
            "eos": False,
            "input_timestamp": self.timestamp,
        }
        self.output_queue.put_nowait(res)
        
        # eos token
        if eos:
            res = {
                "type": "system_audio",
                "data": None,
                "eos": True,
                "input_timestamp": self.timestamp,
            }
            self.output_queue.put_nowait(res)

    async def run(self):
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    res = self.output_queue.get_nowait()
                    await loop.sock_sendall(self.client_socket, length_prefixing(res))
                    if res.get("eos", False):
                        self.mouth.stop()
                except queue.Empty:
                    await asyncio.sleep(0)  # return control to event loop
        except:
            pass
        finally:  # ensure shutdown mouth thread
            self.mouth.stop()
        print("Audio mind closed.")

    def express(self):
        self.mouth.speak()  # starts another thread to chunk data in fixed frequency and put into output queue

    def suspend(self):
        self.mouth.stop()
        if not self.queue.empty() or len(self.chunker.leftover_data) > 0:
            print("[Audio] Mind interrupted.")
            self.queue = queue.Queue()
            self.chunker = Chunker(channel=self.queue)

    async def put(self, x, timestamp):
        # print(f"Put {x}")
        if self.timestamp is None or timestamp != self.timestamp:
            return
        if x.get("eos", False):
            print(f"[Audio] Add to mind: <eos>")
        else:
            print(f"[Audio] Add to mind: ", x["data"].shape)
        self.queue.put(x)
    
    def set_timestamp(self, x):
        self.timestamp = x


class Chunker(object):
    """
    Chunk the binded channel given frame_count.
    Queue is used to ensure thread safe.
    """
    def __init__(self, channel: queue.Queue) -> None:
        self.leftover_data = []
        self.channel = channel

    def gen_block_data(self, frame_count: int, return_true_length=False) -> tuple[np.ndarray, bool]:
        outdata = np.zeros((frame_count,)).astype(np.int16)
        eos = False
        while len(self.leftover_data) < frame_count:
            try:
                more_data = self.channel.get_nowait()
                # eos
                if more_data.get("eos", False):
                    eos = True
                    break
                more_data = more_data["data"]
                self.leftover_data = np.concatenate([self.leftover_data, more_data]) if len(self.leftover_data) else more_data
            except queue.Empty:
                break
        length = min(frame_count, len(self.leftover_data))
        if length > 0:
            if length < frame_count:  # length=0 means silence, and length=frame_count is expected except at the end of the audio
                print(f"underflow: ({length} < {frame_count})")
            outdata[:length] = self.leftover_data[:length]
            self.leftover_data = self.leftover_data[length:]
        if return_true_length:
            return outdata, eos, length
        return outdata, eos


class Mouth(object):
    def __init__(self,
        callback,
        samplerate: int=16000,
        freq: float=0.05,
        id: str="none"
    ) -> None:
        self._open = False
        self.samplerate = samplerate
        self.callback = callback
        self.id = id
        self.freq = freq

        self.thd = None

    def _speak(self):
        t0 = time.perf_counter()    # Time ref point in ms, refer to https://blog.csdn.net/u014147522/article/details/130927851
        time_counter = t0           # Will be incremented with freq for each iteration
        freq = self.freq

        # time_log = []
        while self._open:
            # sleep
            elapsed_time = time.perf_counter() - time_counter
            if elapsed_time < freq:
                target_time =  freq - elapsed_time
                time.sleep(target_time)
            
            # callback
            if time_counter == t0:  # emit a large chunk if first iter
                frame_count = 1600
            else:
                frame_count = int((time.perf_counter() - time_counter) * self.samplerate)
            # self.log(f"Should be triggered at {self.freq * 1e3:.2f}ms, in fact: {(time.perf_counter() - time_counter) * 1e3:.2f}ms.")
            # time_log.append(f"Before cbk ({ii}): {(time.perf_counter() - time_counter) * 1e3:.2f}ms.")
            self.callback(frame_count)
            # time_log.append(f"After cbk ({ii}): {(time.perf_counter() - time_counter) * 1e3:.2f}ms.")
            # self.log(f"#frame: {frame_count}")
            
            time_counter += freq
        # self.log(f"Use: {time.perf_counter() - t0:.2f}s")
    
    def speak(self):
        if self._open:
            return
        # Schedule a task
        self.log("Expression start.")
        self._open = True
        self.thd = threading.Thread(target=self._speak, daemon=True)
        self.thd.start()
    
    def stop(self):
        if not self._open:
            return
        self.log("Expression stop.")
        self._open = False
        self.thd.join()
        self.thd = None

    def log(self, msg: str):
        print(f"[{self.id}]: {msg}")


class AlignedSequence(object):
    """
    Data structure to handle sequence aligned with audio.
    Variable **cur** points to where express stopped.
    """
    def __init__(self):
        self.content = [None]
        self.ends = [0]
        self.cur = 1

    def chunk_to(self, n: int) -> list:
        """
        Chunk when audio has played n samples, this is designed for text/audio output align.
        """
        if self.cur >= len(self.ends):
            return []
        if self.ends[self.cur] > n:
            return []
        ed = self.cur
        while self.ends[ed] <= n:
            ed += 1
            if ed == len(self.ends):
                break
        content_chunk = self.content[self.cur:ed]
        self.cur = ed

        return content_chunk

    def add(self, seq: list, timestamps: list[int]):
        if timestamps is None:
            timestamps = [0.1 * 16000 * (i+1) for i in range(len(seq))]
        last_end = self.ends[-1]
        for (x, t) in zip(seq, timestamps):
            self.content.append(x)
            self.ends.append(last_end + t)


class TextAudioMindHandler(object):
    def __init__(self) -> None:
        self.aligned = AlignedSequence()
        self.audio_queue = queue.Queue()  # audio input queue
        self.output_queue = queue.Queue()  # audio output queue
        self.timestamp = None
        self._is_empty = True

        self.chunker = Chunker(channel=self.audio_queue)
        self.mouth = Mouth(
            callback=self._speak_callback,
            samplerate=16000,
            freq=0.05,
            id="Audio"
        )
        self.mind_state_lock = asyncio.Lock()  # run() contains critical section

        # Alignment
        self.length_expressed = 0
        
        # callbacks
        self.audio_output_callback = None
        self.content_output_callback = None
    
    def _speak_callback(self, frame_count: int):
        data, eos, length = self.chunker.gen_block_data(frame_count, return_true_length=True)
        assert len(data) == frame_count
        res = {
            "data": data,
            "eos": eos,
            "length": length
        }
        self.output_queue.put_nowait(res)

    async def put_content(self, x: dict):
        async with self.mind_state_lock:
            # print(f"[Content] Add to mind: ", x)
            self.aligned.add(seq=x["data"], timestamps=x["timestamps"])
            self._is_empty = False

    async def put_audio(self, x: dict):
        async with self.mind_state_lock:
            if x.get("eos", False):
                print(f"[Audio] Add to mind: <eos>")
            else:
                print(f"[Audio] Add to mind: ", x["data"].shape)
        self.audio_queue.put(x)
        self._is_empty = False
    
    def set_timestamp(self, x):
        self.timestamp = x

    def empty(self) -> bool:
        return self._is_empty
    
    async def run(self):
        assert self.audio_output_callback is not None and self.content_output_callback is not None
        try:
            while True:
                try:
                    async with self.mind_state_lock:
                        res = self.output_queue.get_nowait()
                        await self.audio_output_callback(res["data"], res["eos"])
                        # handle text output aligned by audio output / handle assistant said
                        self.length_expressed += res["length"]
                        content_chunk = self.aligned.chunk_to(self.length_expressed)
                        # if content_chunk:
                            # print(self.length_expressed)
                            # print(content_chunk)
                        await self.content_output_callback(content_chunk, eos=False)
                        if res.get("eos", False):
                            self.mouth.stop()
                            while not self.output_queue.empty():
                                _ = self.output_queue.get_nowait()
                            self.length_expressed = 0
                            await self.content_output_callback([], eos=True)
                except queue.Empty:
                    await asyncio.sleep(0)  # return control to event loop
        except:
            raise
        finally:  # ensure shutdown mouth thread
            self.mouth.stop()
            print("[Text/Audio] Mind closed.")

    def express(self):
        self.mouth.speak()  # starts another thread to chunk data in fixed frequency and put into output queue

    async def suspend(self, new_timestamp):
        async with self.mind_state_lock:
            self.mouth.stop()
            while not self.output_queue.empty():
                _ = self.output_queue.get_nowait()
            self.length_expressed = 0
            if not self.empty():
                self.aligned = AlignedSequence()
                self.audio_queue = queue.Queue()
                self.chunker = Chunker(channel=self.audio_queue)
                print("[Text/Audio] Mind interrupted.")
        self.set_timestamp(new_timestamp)
        self._is_empty = True
