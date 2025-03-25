import os
import numpy as np
import librosa

from openai import AsyncOpenAI

from server.utils import wav_normalization


class AsyncModelClient(object):
    def __init__(self, model_name, language_code="en-US", sr=16000) -> None:
        self._client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model_name = model_name
        self.language_code = language_code
        self.sr = sr
    
    def sample_rate(self):
        return self.sr

    def set_speaker(self, speaker: str):
        self.voice_name = speaker
    
    async def text_to_wav(self, text: str, voice_name="alloy", config={}) -> np.ndarray:
        """ return wav numpy array in int16 """
        response = await self._client.audio.speech.create(
            model=self.model_name,
            voice=voice_name,
            input=text,
            response_format="wav",
        )

        audio_bytes = b""
        for data in response.response.iter_bytes():
            audio_bytes += data
        wav = np.frombuffer(audio_bytes, dtype=np.int16)
        # print(wav.shape)
        # wavfile.write("check0.wav", 24000, wav)

        # resample
        wav = wav.astype(np.float32) / 32768
        wav = librosa.resample(wav, orig_sr=24000, target_sr=self.sr)
        wav = wav_normalization(wav)
        wav = (wav * 32000).astype(np.int16)
        
        return wav

    async def close(self):
        await self._client.close()