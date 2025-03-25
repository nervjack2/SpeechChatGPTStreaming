import numpy as np
import time

from server.utils import Client


class SyncClient(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port


class AsyncClient(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._client = Client(host, port)

        self.recv_stream = self._client.recv_stream
    
    async def run(self):
        try:
            await self._client.run()
        except Exception as e:
            print("Failed to connect to server!")
            raise
    
    def _wrap_system_prompt(self, data: str) -> dict:
        return {
            "type": "system_prompt",
            "data": data,
            "input_timestamp": time.time(),
        }
    
    def _wrap_user_text(self, data: str) -> dict:
        return {
            "type": "user_text",
            "data": data,
            "input_timestamp": time.time(),
        }

    def _wrap_audio(self, data: np.ndarray) -> dict:
        return {
            "type": "audio",
            "data": data,
            "input_timestamp": time.time(),
        }

    async def send_system_prompt(self, data: str):
        await self.send_data(self._wrap_system_prompt(data))

    async def send_user_text(self, data: str):
        await self.send_data(self._wrap_user_text(data))

    async def send_audio(self, data: np.ndarray):
        await self.send_data(self._wrap_audio(data))

    # async def send_assistant_said(text, audio)
    #     await self.send_data({
    #         "type": "assitant_said",
    #         "text_data": text,
    #         "audio_data": audio,
    #         "input_timestamp": time.time(),
    #     })

    async def send_data(self, data):
        await self._client.send_data(data)

    async def reset(self):
        await self.send_user_text("===")

    async def close(self):
        await self._client.close()
