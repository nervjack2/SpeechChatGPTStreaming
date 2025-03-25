import asyncio
import queue
import time


class DynamicChunker(object):
    def __init__(self) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.latency = 0.2
        self.real_tps = 50
        self.foward_rtf = 0.05
        self.min_chunk_size = 10

        self.t_init = None
        self.t_chunk = None
        self.t_audio_end = None

        self.concat_data = []
        self.concat_data_timestamp = None
        self.eos = False
    
    def reset(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.t_init = None
        self.t_chunk = None
        self.t_audio_end = None

        self.concat_data = []
        self.concat_data_timestamp = None
        self.eos = False

    def _init_timestamps(self):
        self.t_init = time.time()
        self.t_audio_end = self.latency
        self._preserve_forward_time()

    def _get_timestamp(self) -> float:
        return time.time() - self.t_init
    
    def _preserve_forward_time(self):
        preserve_ratio = 0.5
        self.t_preserved = min(0.3, (self.t_audio_end - self._get_timestamp()) * preserve_ratio)
        self.t_chunk = self.t_audio_end - self.t_preserved
        self.max_chunk_size = max(self.min_chunk_size, int(self.t_preserved / self.foward_rtf * self.real_tps))
        print(f"next end: {self.t_audio_end:.2f}, next chunk time: {self.t_chunk:.2f}, max chunk size: {self.max_chunk_size:.2f}")
    
    def _chunk(self):
        if len(self.concat_data) >= self.min_chunk_size or self.eos:
            if len(self.concat_data) > 0:
                chunk = self.concat_data[:self.max_chunk_size]
                self.concat_data = self.concat_data[self.max_chunk_size:]
                self.output_queue.put({
                    "type": "system_token",
                    "data": chunk,
                    "eos": False,
                    "input_timestamp": self.concat_data_timestamp,
                })
                self.t_audio_end += len(chunk) / self.real_tps
                self._preserve_forward_time()

            if self.eos and len(self.concat_data) == 0:  # completed
                self.output_queue.put({
                    "type": "system_audio",
                    "data": None,
                    "eos": True,
                    "input_timestamp": self.concat_data_timestamp,
                })
                self.t_init = None
                self.eos = False
                self.concat_data = []
                self.concat_data_timestamp = None
        else:  # chunk too short and response not ended, increase t_chunk
            self.t_audio_end += self.min_chunk_size / self.real_tps
            self._preserve_forward_time()

    async def run(self):
        while True:
            if self.t_init is not None and self._get_timestamp() > self.t_chunk:
                print(f"Now: {self._get_timestamp():.2f} vs Expected: {self.t_chunk:.2f} vs End: {self.t_audio_end:.2f}")
                self._chunk()
            await asyncio.sleep(0.01)
    
    def input_data(self, res):
        # deal with mind interruption, chunks should have the latest timestamp
        if res['input_timestamp'] != self.concat_data_timestamp:
            self.concat_data_timestamp = res['input_timestamp']
            self.concat_data = []
            self._init_timestamps()

        if res["data"] is not None:  # update system text if not blank (eos is considered blank)
            self.eos = False
            self.concat_data += res["data"]

        # handle system text eos
        if res.get("eos", False):
            self.eos = True
