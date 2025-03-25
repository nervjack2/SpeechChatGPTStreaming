import importlib.util
import os
import sys
import typing


# New mapping with dynamic module import
ASR_SERVER_MAPPING = {
    "whisper-api": ("server/Whisper/processor.py", "WhisperServerProcessor"),
    "whisper": ("server/Whisper/processor.py", "WhisperServerProcessor"),
}

SPEECH_UNIT_SERVER_MAPPING = {
    "streaming-hubert-discrete": ("server/StreamingHuBERTDiscrete/processor.py", "StreamingHuBERTDiscreteServerProcessor"),
}

LLM_SERVER_MAPPING = {
    "gpt": ("server/GPT/processor.py", "GPTServerProcessor"),
    "slm-v0": ("server/SpeechLM/processor_native_pytorch.py", "SpeechLMServerProcessor"),
    "slm-v0.1": ("server/SpeechLM/processor_native_pytorch2.py", "SpeechLMServerProcessor"),  # use AsyncModelClient API
    "slm-v1": ("server/SpeechLM/processor_native_pytorch.py", "SpeechLMServerProcessor"),
    "slm-v1.1": ("server/SpeechLM/processor_native_pytorch2.py", "SpeechLMServerProcessor"),  # use AsyncModelClient API
    "slm-v1.2": ("server/SpeechLM/processor_native_pytorch3.py", "SpeechLMServerProcessor"),  # use interleaving
    "slm": ("server/SpeechLM/processor_native_pytorch3.py", "SpeechLMServerProcessor"),
    
    # trt version
    "slm-trt-v0": ("server/SpeechLM/processor_tensorrt.py", "SpeechLMServerProcessor"),
    "slm-trt-v0.1": ("server/SpeechLM/processor_tensorrt2.py", "SpeechLMServerProcessor"),
    "slm-trt-v1": ("server/SpeechLM/processor_tensorrt.py", "SpeechLMServerProcessor"),
    "slm-trt-v1.1": ("server/SpeechLM/processor_tensorrt2.py", "SpeechLMServerProcessor"),
    "slm-trt-v1.2": ("server/SpeechLM/processor_tensorrt3.py", "SpeechLMServerProcessor"),
    "slm-trt": ("server/SpeechLM/processor_tensorrt3.py", "SpeechLMServerProcessor"),
}

TTS_SERVER_MAPPING = {
    "openai-tts": ("server/TTS/_processor_openai.py", "OpenAITTSServerProcessor"),
    "tts": ("server/TTS/processor_openai.py", "OpenAITTSServerProcessor"),
}

VOC_SERVER_MAPPING = {
    "voc": ("server/Vocoder/processor_simple.py", "VocoderServerProcessor"),
    "voc-diffusion": ("server/Vocoder/processor_diffusion.py", "VocoderServerProcessor"),
}

MAIN_SERVER_MAPPING = {
    "client": ("server/Client/web_server.py", "WebRTCProcessor"),
    "main": ("server/SpeechChatGPT/processor2.py", "MainServerProcessor"),
    "main-api": ("server/CascadeAPI/processor.py", "MainServerProcessor"),
}


SERVER_MAPPING = {
    **ASR_SERVER_MAPPING,
    **SPEECH_UNIT_SERVER_MAPPING,
    **LLM_SERVER_MAPPING,
    **TTS_SERVER_MAPPING,
    **VOC_SERVER_MAPPING,
    **MAIN_SERVER_MAPPING,
}


def get_class_in_module(class_name: str, module_path: typing.Union[str, os.PathLike]) -> typing.Type:
    """
    From huggingface. Import a module and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_spec = importlib.util.spec_from_file_location(name, location=module_path)
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(module_spec)
        # insert it into sys.modules before any loading begins
        sys.modules[name] = module
    # reload in both cases
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)


def get_server_cls(name):
    module_path, class_name = SERVER_MAPPING[name]
    return get_class_in_module(class_name, module_path)
