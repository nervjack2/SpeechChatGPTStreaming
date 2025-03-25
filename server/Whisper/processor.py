import socket
import asyncio
import numpy as np
from threading import Thread
import queue

from server.common.template import DefaultProcessor, DefaultServerProcessor
from .whisper_online import *
from .whisper_utils import *


class StreamWhisper(object):
    SAMPLING_RATE = 16000
    input_queue: queue.Queue
    output_queue: queue.Queue

    def __init__(self, config) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.audio_chunks = []
        self.input_timestamp = None
        self.last_end = None

        self.min_chunk = config["min-chunk-size"]
        _, self.online_asr_proc = asr_factory_from_config(config)
        self.online_asr_proc.init()
        self._warmup()

    def _warmup(self):
        warmup_audio_len = int(self.min_chunk*self.SAMPLING_RATE)
        warmup_audio = np.random.rand(warmup_audio_len, 1).astype(np.float32)
        self.online_asr_proc.insert_audio_chunk(warmup_audio)
        trans, seg = self.online_asr_proc.process_iter()
        self.online_asr_proc.init()
        print("Warmup complete.")
    
    # output parsing
    def format_output_transcript(self, o):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if o[0] is not None:
            beg, end = o[0]*1000,o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            # print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
            if not check_hallucination(o[2]):
                return {
                    "beg": int(beg),
                    "end": int(end),
                    "data": o[2],
                }
            else:
                return {
                    "beg": int(beg),
                    "end": int(end),
                    "data": "===",
                }
        else:
            return None
        
    # handler functions
    def _handle_reset(self, res):
        self.online_asr_proc.init()
        print("========== reset all state ==========")

    def _handle_audio(self, res):
        self.audio_chunks.append(res["data"])
        if self.input_timestamp is None:
            self.input_timestamp = res["input_timestamp"]
        if sum(len(x) for x in self.audio_chunks) < self.min_chunk * self.SAMPLING_RATE:
            return
        concat_audio = np.concatenate(self.audio_chunks)
        # self.accu_len += len(concat_audio)
        self.audio_chunks.clear()
        input_timestamp = self.input_timestamp
        self.input_timestamp = None

        st = time.time()
        self.online_asr_proc.insert_audio_chunk(concat_audio)
        trans, seg = self.online_asr_proc.process_iter()
        print("Confirmed: ", seg)
        print(f"Forward: {time.time()-st:.2f}s.")
        
        data = self.format_output_transcript(trans)
        if data is None:
            return
        if data["data"] == '===':
            return_type = "hallucinate"
        else:
            return_type = "user_text"
        res = {
            "type": return_type,
            "data": data["data"],
            "input_timestamp": input_timestamp,
            # "accu_len": self.accu_len,
            "segment": seg,
        }
        # self.accu_len = 0
        self.output_queue.put(res)

    def run(self):
        """ loop to process input queue """
        while True:
            if self.input_queue.empty():
                continue
            else:
                res = self.input_queue.get()

            # update assistant history
            if res["type"] == "audio":  # update assistant history
                self._handle_audio(res)
            elif res["type"] == "reset":  # reset signal
                self._handle_reset(res)
            else:
                raise NotImplementedError


class WhisperServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self) -> None:
        print("Run Stream Whisper on new thread!")
        self.model = StreamWhisper(self.config)
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> DefaultProcessor:
        p = DefaultProcessor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
