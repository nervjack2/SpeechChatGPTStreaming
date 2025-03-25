import sounddevice as sd
import threading
import queue
import numpy as np


# Define audio stream parameters
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000


# class AudioStreamPlayerOld(object):
#     def __init__(self) -> None:
#         self.audio_buffer = queue.Queue()

#         # Create a PyAudio instance
#         self.p = pyaudio.PyAudio()

#         # Open a PyAudio stream
#         self.stream = self.p.open(
#             output_device_index=3,
#             format=FORMAT,
#             channels=CHANNELS,
#             rate=RATE,
#             output=True,
#             frames_per_buffer=CHUNK,
#         )

#     def start_stream(self):
#         threading.Thread(target=self.play, daemon=True).start()

#     def play(self):
#         while True:
#             audio_bytes = self.audio_buffer.get()
#             self.stream.write(audio_bytes)
    
#     def write(self, data: np.ndarray):
#         self.audio_buffer.put(data.tobytes())

#     def close(self):
#         self.stream.stop_stream()
#         self.stream.close()
#         self.p.terminate()


class Chunker(object):
    """
    Chunk the binded channel given frame_count.
    Queue is used to ensure thread safe.
    """
    def __init__(self, channel: queue.Queue) -> None:
        self.leftover_data = []
        self.channel = channel

    def gen_block_data(self, frame_count: int) -> np.ndarray:
        outdata = np.zeros((frame_count,)).astype(np.float32)
        while len(self.leftover_data) < frame_count:
            try:
                more_data = self.channel.get_nowait()
                self.leftover_data = np.concatenate([self.leftover_data, more_data]) if len(self.leftover_data) else more_data
            except queue.Empty:
                break
        length = min(frame_count, len(self.leftover_data))
        if length > 0:
            if length < frame_count:  # length=0 means silence, and length=frame_count is expected except at the end of the audio
                print(f"underflow: ({length} < {frame_count})")
            outdata[:length] = self.leftover_data[:length]
            self.leftover_data = self.leftover_data[length:]
        return outdata
    

class AudioStreamPlayer(object):
    """
    Play audio to user from audio buffer using sd.OutputStream
    """
    def __init__(self, samplerate=16000, channels=1):
        self.audio_buffer = queue.Queue()  # Audio buffer (queue or similar structure)
        self.chunker = Chunker(self.audio_buffer)
        self.samplerate = samplerate
        self.channels = channels
        self.stream = None
    
    def put(self, x: np.ndarray):
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768  # sounddevice use range (-1, 1)
        self.audio_buffer.put_nowait(x)

    def callback(self, outdata, frames, time, status):
        # if status:
        #     print(f'Streaming error: {status}')
        data = self.chunker.gen_block_data(frames)
        outdata[:, 0] = data

    def start_stream(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=0,
            callback=self.callback
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.audio_buffer = queue.Queue()
            self.chunker = Chunker(self.audio_buffer)
