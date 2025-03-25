import asyncio
import numpy as np
import queue
import librosa

from client._client import AsyncClient
from client.audio_stream import AudioStreamPlayer
from server.SpeechChatGPT.utils import Mouth
from server.utils import wav_normalization


HOST = "localhost"
PORT = 43007
player = AudioStreamPlayer()


class Wav(object):
    def __init__(self, wav) -> None:
        self.wav = wav


async def receive_user_text(res):
    if res.get("eos", False):
        x = "<eos>"
    else:
        x = res["data"]
    print("User text: ", x)


async def receive_system_text(res):
    if res.get("eos", False):
        x = "<eos>"
    else:
        x = res["data"]
    print("System text: ", x)


async def receive_system_audio(res):
    if res.get("eos", False):
        print("Audio end.")
    else:
        player.put(res["data"])


async def listen_for_server_data(client: AsyncClient):
    try:
        stream = client._client.recv_stream()
        async for res in stream:
            if res["type"] == "system_audio":
                await receive_system_audio(res)
                if res.get("eos", False):
                    return
            elif res["type"] == "user_text":
                await receive_user_text(res)
            elif res["type"] == "system_text":
                await receive_system_text(res)
            else:
                print(f"Unknown message type: {res}")
                raise NotImplementedError
    except (ConnectionResetError, BrokenPipeError):
        print("Client connection closed")
        raise
    except Exception as e:
        raise


async def simulate_microphone_sending(client: AsyncClient, wav: np.ndarray):
    output_queue = queue.Queue()
    wav_obj = Wav(wav)
    async def send():
        while True:
            try:
                res = output_queue.get_nowait()
                if isinstance(res, dict)and res.get("eos", False):
                    break
                await client.send_audio(res)
            except queue.Empty:
                await asyncio.sleep(0)  # return control to event loop

    def speaking_callback(frame_count: int):
        outdata = np.zeros((frame_count,)).astype(np.float32)
        length = min(frame_count, len(wav_obj.wav))
        if length > 0:
            outdata[:length] = wav_obj.wav[:length]
            wav_obj.wav = wav_obj.wav[length:]
            print(outdata.shape)
        elif length == 0:
            outdata = {"eos": True}
        output_queue.put_nowait(outdata)

    fut = asyncio.create_task(send())
    mouth = Mouth(speaking_callback, freq=0.05)
    mouth.speak()
    await fut
    mouth.stop()
        

async def main():
    try:
        client = AsyncClient(HOST, PORT)
        await client.run()
        player.start_stream()
        task = asyncio.create_task(listen_for_server_data(client))
        
        # wav
        audio_path = "_data/test-0.wav"
        wav, _ = librosa.load(audio_path, sr=16000)
        wav = wav_normalization(wav)
        await simulate_microphone_sending(client, wav)

        # text
        # client.send_text("Hello, how are you?")

        await task
        await asyncio.sleep(1)  # Wait for audio to play
    except:
        raise
    finally:
        await client.close()
        player.stop_stream()


if __name__ == "__main__":
    asyncio.run(main())
