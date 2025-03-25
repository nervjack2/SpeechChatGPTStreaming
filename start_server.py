import argparse
import socket
import asyncio
import yaml

from server import Define
from server.load import get_server_cls
from server.utils import handle_asyncio_exception_new


def create_config(args):
    config = None
    if args.name == "gpt":
        config = yaml.load(open("server/GPT/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "main":
        config = yaml.load(open("server/SpeechChatGPT/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "main-exp":
        config = yaml.load(open("server/SpeechChatGPT/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "main-api":
        config = yaml.load(open("server/CascadeAPI/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "whisper":
        config = yaml.load(open("server/Whisper/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "whisper-api":
        config = yaml.load(open("server/Whisper/config-api.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == 'streaming-hubert-discrete':
        config = yaml.load(open("server/StreamingHuBERTDiscrete/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name in ["tts", "openai-tts"]:
        config = yaml.load(open("server/TTS/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name in ["voc"]:
        config = yaml.load(open("server/Vocoder/config.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "voc-diffusion":
        config = yaml.load(open("server/Vocoder/config-diffusion.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name in ["slm", "slm-trt"]:
        config = yaml.load(open("server/SpeechLM/config-taipei1.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name.startswith("slm"):  # different version
        if "v0" in args.name:
            config = yaml.load(open("server/SpeechLM/config.yaml", "r"), Loader=yaml.FullLoader)
        elif "v1" in args.name:
            config = yaml.load(open("server/SpeechLM/config-taipei1.yaml", "r"), Loader=yaml.FullLoader)
    elif args.name == "client":
        config = yaml.load(open("server/Client/config.yaml", "r"), Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unknown tag: {args.name}")
    
    if config is None:
        raise ValueError(f"Unknown tag: {args.name}!")

    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
        
    return config


async def start_server(args):
    config = create_config(args)
    print(config)

    server_cls = get_server_cls(args.name)
    processor = server_cls(config=config)

    host = config["host"]
    port = config["port"]
    max_conn = config["max_conn"]
    
    # Step 1: Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Step 2: Bind the socket to an address and port
    server_socket.bind((host, port))

    server_socket.setblocking(False)
    loop = asyncio.get_event_loop()
    
    # Step 3: Listen for incoming connections
    server_socket.listen(max_conn)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = await loop.sock_accept(server_socket)
        print(f"Got connection from {addr}")
        task = asyncio.create_task(processor.process(client_socket, addr))
        task.add_done_callback(handle_asyncio_exception_new)  # error should be handled properly inside task, no error should be raised here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-h', '--host', required=False)
    parser.add_argument('-p', '--port', type=int, required=False)
    parser.add_argument('--debug', type=str, default=None)
    args = parser.parse_args()

    if args.debug is not None:
        Define.DEBUG_MODE = args.debug

    if args.name == "client":
        config = create_config(args)
        print(config)

        server_cls = get_server_cls(args.name)
        processor = server_cls(config=config)

        processor.run_webrtc_server()
    else:
        asyncio.run(start_server(args))
