import os
import asyncio
import numpy as np
import pickle
import time
import wave
import traceback


def length_prefixing(response):
    serialized_response = pickle.dumps(response)
    response_length = len(serialized_response)
    response_header = response_length.to_bytes(4, 'big')
    return response_header + serialized_response


def wav_normalization(wav: np.array) -> np.array:
    denom = max(abs(wav))
    if denom == 0 or np.isnan(denom):
        raise ValueError
    return wav / denom


async def recv_with_length_prefixing(client_socket=None, reader=None):
    if not client_socket and not reader:
        raise ValueError
    if client_socket and reader:
        raise ValueError
    if client_socket:
        loop = asyncio.get_event_loop()
        header = await loop.sock_recv(client_socket, 4)
    else:
        header = await reader.read(4)
    if not header:
        return header
    
    # recv with specified length
    res_length = int.from_bytes(header, 'big')
    data = bytearray()
    while len(data) < res_length:
        if client_socket:
            res = await loop.sock_recv(client_socket, res_length - len(data))
        else:
            res = await reader.read(res_length - len(data))
        if not res:
            raise ConnectionError("Socket connection lost")
        data.extend(res)
    
    return data


class Client(object):
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.reader, self.writer = None, None
        
        self._no_send = False
    
    async def run(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
    
    async def send_data(self, data: dict):
        if self._no_send:
            return
        assert self.writer is not None, "Please call run() to build connection before calling."
        data["timestamp"] = time.time()  # add sending time
        self.writer.write(length_prefixing(data))
        await self.writer.drain()

    async def recv_stream(self):
        assert self.reader is not None, "Please call run() to build connection before calling."
        while True:
            res = await recv_with_length_prefixing(reader=self.reader)
            if not res:
                print("Server closed the connection.")
                break
            res = pickle.loads(res)
            yield res

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()

    def enable(self):
        self._no_send = False
    
    def disable(self):
        """ disable send_data for debug """
        self._no_send = True


class AudioLogger(object):

    # Parameters for audio recording
    CHANNELS = 1  # 1 channel for mono, 2 for stereo
    RATE = 16000  # Sample rate (samples per second)
    SAMPLE_WIDTH = 2  # Sample width in bytes (2 bytes for 16-bit audio)
    
    def __init__(self, output_filename: str) -> None:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Open WAV file for writing
        wavefile = wave.open(output_filename, 'wb')
        wavefile.setnchannels(AudioLogger.CHANNELS)
        wavefile.setsampwidth(AudioLogger.SAMPLE_WIDTH)
        wavefile.setframerate(AudioLogger.RATE)
        self.wavefile = wavefile
        self.output_filename = output_filename
        self.stopped = False

    def write(self, res: dict):
        if not self.stopped:
            audio_data = res["data"]
            # print(max(audio_data))
            byte_data = audio_data.tobytes()
            self.wavefile.writeframes(byte_data)

    def stop(self):
        if not self.stopped:
            self.stopped = True
            print(f"Audio data saved to {self.output_filename}.")
            self.wavefile.close()


def handle_asyncio_exception(future):
    # Ensure we catch the exception from couroutine even on a different thread
    try:
        result = future.result()
    except Exception as e:
        raise e


# callback func called for all tasks
def handle_asyncio_exception_new(fut: asyncio.Task):
    # check if the task had an exception
    if fut.exception():
        try:
            # re-raise exception
            fut.result()
        except Exception as e:
            traceback.print_exc()


async def run_parallel_tasks(tasks: list[asyncio.Task]):
    """ Stop all tasks if any of them was successful or raised an exception. Exception is handled. """
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        handle_asyncio_exception_new(task)

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
