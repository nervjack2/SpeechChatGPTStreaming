import numpy as np
import queue
import threading
import time


class Chunker(object):
    """
    Chunk the binded channel given frame_count.
    Queue is used to ensure thread safe.
    """
    def __init__(self, channel: queue.Queue) -> None:
        self.leftover_data = []
        self.channel = channel

    def gen_block_data(self, frame_count: int) -> tuple[np.ndarray, bool]:
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
