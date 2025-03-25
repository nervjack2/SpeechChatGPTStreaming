# web_server.py
import asyncio
import pickle
import time
import queue
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from .. import Define
from ..utils import length_prefixing, recv_with_length_prefixing, Client, AudioLogger, handle_asyncio_exception, run_parallel_tasks

from aiohttp import web
import aiohttp_cors
import aiohttp
import aiofiles
import logging
import json
import wave
from pydub import AudioSegment  # Requires ffmpeg to be installed and in the PATH
import av
import threading
import traceback
import yaml
import os
import ssl
    
class WebRTCMediaHandler(MediaStreamTrack):
    """ Custom MediaStreamTrack to handle audio streaming. """
    kind = "audio"

    def __init__(self, track, client):
        super().__init__()  # Initialize base MediaStreamTrack
        self.track = track
        self.client = client
        self.wav_file = None  # Will be initialized when the first frame is received
        self.processed_wav = None # Will be initialized when the first frame is received
        self.frame_data_queue = queue.Queue()

    async def recv(self):
        try:
            frame = await self.track.recv()
            if self.client.mic_on:
            #     print('mic on')
                await self.process_and_send_frame(frame)
            # await self.save_frame_to_wav(frame)
            # return frame
            # print("called recv")
            new_frame = await self.client.audio_mind_handler.get_next_frame(frame)
            # print(f"Received frame from audio mind: {new_frame}")
            return new_frame
        except Exception as e:
            print(f"Failed to receive frame: {e}")

    async def process_and_send_frame(self, frame):
        try:
            audio_array = frame.to_ndarray()

            # Process the audio frame (resample to 16kHz, mono, s16_LE format)
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=frame.rate,
                sample_width=frame.format.bytes,
                channels=len(frame.layout.channels)
            )
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

            audio_array = np.array(audio_segment.get_array_of_samples())

            data = {
                "type": "audio",
                "data": audio_array.reshape(-1, 1),  # Convert to 2D array (n_samples, 1)
                "input_timestamp": time.time(),
            }
            await self.client.send_data(data)
            # print("Sent frame to server via socket.")

        except Exception as e:
            raise e

    def initialize_wav_file(self, frame):
        """Initialize the WAV file with the correct parameters based on the first frame."""
        self.wav_file = wave.open('/Users/huenpei/Desktop/NTU/r2_2/ML_codespace/SpeechChatGPTStreaming/_data/output.wav', 'wb')
        self.wav_file.setnchannels(len(frame.layout.channels))  # Number of channels
        # Sample width (bytes) is determined by the format (e.g., 's16le' -> 2 bytes)
        bytes_per_sample = frame.format.bytes
        print(f"bytes_per_sample: {bytes_per_sample}")
        self.wav_file.setsampwidth(bytes_per_sample)
        self.wav_file.setframerate(frame.rate)

        self.processed_wav = wave.open('/Users/huenpei/Desktop/NTU/r2_2/ML_codespace/SpeechChatGPTStreaming/_data/output_processed.wav', 'wb')
        self.processed_wav.setnchannels(1)
        self.processed_wav.setsampwidth(2)
        self.processed_wav.setframerate(16000)

    async def save_frame_to_wav(self, frame):
        try:
            if self.wav_file is None:
                # Initialize the WAV file when the first frame is received
                self.initialize_wav_file(frame)

            # Convert the audio frame to a NumPy array and write to WAV file
            audio_array = frame.to_ndarray()
            # print(f"audio_array: {audio_array.shape}")
            self.wav_file.writeframes(audio_array.tobytes())
            # print("Saved frame to output.wav")

            # Write to processed WAV file (resampled to 16kHz, mono, s16_LE format)
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=frame.rate,
                sample_width=frame.format.bytes,
                channels=len(frame.layout.channels)
            )
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

            # Save the processed frame to a WAV file
            self.processed_wav.writeframes(audio_segment.raw_data)

            # print("Saved frame to output_processed.wav")
        except Exception as e:
            print(f"Failed to save frame to WAV: {e}")

class AudioMindHandler(object):
    def __init__(self) -> None:
        self.queue = queue.Queue()
        self.frame_data_queue = queue.Queue()

        self.running_status = True

    def gen_block_data(self, leftover_data: np.ndarray, block_size: int):
        """ return a fix-sized audio numpy array and eos flag """
        frames_to_read = block_size
        data = []
        length = 0
        eos = False
        while length < frames_to_read:
            if len(leftover_data) >= frames_to_read - length:
                # We have enough leftover data to fulfill the request
                data.append(leftover_data[:frames_to_read - length])
                leftover_data = leftover_data[frames_to_read - length:] if len(leftover_data) > frames_to_read - length else []
                length = frames_to_read
            else:
                if not self.queue.empty():
                    # Get more data from the queue
                    more_data = self.queue.get()
                    # print(more_data)
                    if more_data.get("eos", False):
                        eos = True
                        data.append(np.zeros((frames_to_read - length,)).astype(np.int16))
                        break
                    audio_data = more_data["data"]
                    stereo_data = self.convert_to_stereo(audio_data)
                    # print(stereo_data.shape)
                    # leftover_data = np.concatenate([leftover_data, audio_data]) if len(leftover_data) else audio_data
                    leftover_data = np.concatenate([leftover_data, stereo_data]) if len(leftover_data) else stereo_data
                else:
                    # data.append(np.zeros((frames_to_read - length,)).astype(np.int16))
                    break
        return data, eos, leftover_data

    async def run(self):
        leftover_data = []
        # frames_to_read = 4000
        frames_to_read = 1920
        try:
            while self.running_status:  # loops forever
                data, eos, leftover_data = self.gen_block_data(leftover_data, block_size=frames_to_read)
                if len(data):
                    data = np.concatenate(data)
                    assert len(data) == frames_to_read, f"make sure transmit with fix size! data shape: {data.shape}"
                    self.frame_data_queue.put(data)
                    # print(f"Put frame to output queue: {data.shape}")
                await asyncio.sleep(0)  # give control to event loop
        except Exception as e:
            print(f'Failed when running audio mind: {e}')
        print("Audio mind closed.")

    def convert_to_stereo(self, data):
        try:
            # First use audio segment to convert to audio frame
            audio_segment = AudioSegment(
                data.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            # Convert to 48000, 2 channels, s16_LE format
            audio_segment = audio_segment.set_channels(2).set_frame_rate(48000)

            # Convert the modified audio segment back to numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

            return samples

        except Exception as e:
            print(f"Failed to convert to stereo: {e}")

    def convert_to_frame(self, sample, frame):
        try:
            sample = sample.reshape(1, -1)
            # # Create an av.AudioFrame from the numpy array
            new_frame = av.AudioFrame.from_ndarray(sample, format='s16', layout=frame.layout)
            # Set all the settings as the original frame
            for attr in ["sample_rate", "time_base", "pts", "dts"]:
                setattr(new_frame, attr, getattr(frame, attr))
            return new_frame
        except Exception as e:
            print(f"Failed to convert to frame: {e}")

    async def get_next_frame(self, frame):
        try:
            if self.frame_data_queue.empty():
                # return silence frame with the same format
                frame_shape = frame.to_ndarray().shape
                silence_frame = frame.from_ndarray(np.zeros(frame_shape, dtype=np.int16), format='s16', layout=frame.layout)
                # set all the setting as original frame
                for attr in ["sample_rate", "time_base", "pts", "dts"]:
                    setattr(silence_frame, attr, getattr(frame, attr))
                return silence_frame
            new_frame_data = self.frame_data_queue.get()
            # print("New frame data: ", new_frame_data)
            new_frame = self.convert_to_frame(new_frame_data, frame)
            # print(f"Received frame from input: {frame}")
            # print(f"Received frame from audio mind: {new_frame}")
            return new_frame
        except Exception as e:
            print(f"Failed to get next frame: {e}")
           

class SpeechChatGPTClient(object):
    """ Class to handle the client-side of the SpeechChatGPT system. """
    def __init__(self, host, port, ws):
        self.local_track = None
        self.system_audio_playing = False
        self.system_text_showing = False
        self.ws = ws

        # client wrapper
        self.client = Client(host, port)

        self.audio_mind_handler = AudioMindHandler()

        self.mic_on = True

        # self.listen_for_server_data()
        print(f"SpeechChatGPT client {host}:{port} initialized.")

    async def connect_to_client(self):
        try:
            await self.client.run()
        except Exception as e:
            print(f"Failed to connect to client: {e}")

    # add audio track
    def add_audio_track(self, track):
        self.local_track = WebRTCMediaHandler(track, self)
        return self.local_track

    # handle input
    async def send_data(self, data):
        if data["type"] != "audio":
            print(f"Send: {data}")
        await self.client.send_data(data)


    async def listen_for_server_data(self):
        try:
            stream = self.client.recv_stream()
            async for res in stream:
                if res["type"] == "system_audio":
                    await self.receive_system_audio(res)
                elif res["type"] == "user_text":
                    await self.receive_user_text(res)
                elif res["type"] == "system_text":
                    await self.receive_system_text(res)
                elif res["type"] == "command":
                    await self.receive_command(res)
                else:
                    print(f"Unknown message type: {res}")
                    raise NotImplementedError
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise

    # handle output
    async def receive_system_audio(self, res):
        # print(res)
        if res.get("eos", False):
            print("system audio break")
            self.system_audio_playing = False
            return
        if self.local_track:
            self.audio_mind_handler.queue.put(res)
        if not self.system_audio_playing:
            if max(res["data"]) > 0:  # first non-empty chunk
                print(f"ASR + LLM + TTS latency: {time.time()-res['input_timestamp']:.2f}s.")
                self.system_audio_playing = True
        # self.audio_stream_player.write(res["data"])
    
    async def receive_user_text(self, res):
        # print(res)
        if res.get("eos", False):
            print("user break")
            return
        await self.ws.send_json(res)
        print(f"ASR latency: {time.time()-res['input_timestamp']:.2f}s.")

    async def receive_system_text(self, res):
        # print(res)
        if res.get("eos", False):
            print("system break")
            self.system_text_showing = False
            return
        await self.ws.send_json(res)
        if not self.system_text_showing:
            # print(f"LLM latency: {time.time()-res['turn_take_timestamp']:.2f}s.")
            print(f"ASR + LLM latency: {time.time()-res['input_timestamp']:.2f}s.")
            self.system_text_showing = True

    async def receive_command(self, res):
        await self.ws.send_json(res)
        print(f"Received command: {res}")

    async def close(self):
        print("Closing SpeechChatGPT client...")
        # remove the local track
        if self.local_track:
            del self.local_track
        await self.client.close()

    # handle convert
    def convert_to_stereo(self, data):
        try:
            # First use audio segment to convert to audio frame
            audio_segment = AudioSegment(
                data.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            # Convert to 48000, 2 channels, s16_LE format
            audio_segment = audio_segment.set_channels(2).set_frame_rate(48000)

            # Convert the modified audio segment back to numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

            return samples

        except Exception as e:
            print(f"Failed to convert to stereo: {e}")

class WebRTCProcessor(object):
    """ Class to handle WebRTC connections and media. """
    def __init__(self, config):
        self.logger = logging.getLogger("WebRTCProcessor")
        self.logger.setLevel(logging.DEBUG if Define.DEBUG_MODE else logging.INFO)

        # Create a console handler and set the level to DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Add the handler to the logger
        self.logger.addHandler(ch)


        self.pcs = set()
        self.config = config

        self.running_status = True

        self.model_nodes = {}
        self.model_status = []
        self.load_model_nodes()

        self.rooms = {} # Key: roomId, Value: set of clients

        self.model_status_clients = {} 

    async def check_node_status(self, node_host, node_port):
        """ Check the status of a model node. """
        status = "off"
        # try connect the node
        try:
            client = Client(node_host, node_port)
            await client.run()
            status = "on"
            await client.close()
        except Exception as e:
            print(f"Failed to connect to model node: {e}")
        return status

    def load_model_nodes(self):
        # Read the model nodes from the config file
        for node_id, node_info in self.config.items():
            node_name = node_info.get('name')
            node_host = node_info.get('host')
            node_port = node_info.get('port')
            max_conn = node_info.get('max_conn')
            node_status = asyncio.run(self.check_node_status(node_host, node_port))
            cur_conn = 0

            node = {
                'name': node_name,
                'host': node_host,
                'port': node_port,
                'max_conn': max_conn,
                'status': node_status,
                'cur_conn': cur_conn
            }
            self.model_nodes[node_name] = node
            self.model_status.append({'name': node_name, 'status': node_status, 'full': False})

        self.logger.info(f"Loaded {len(self.model_nodes)} model nodes")

    async def check_model_status(self, request):
        """
        Return: if status changed, return true, else return false
        """
        print("Getting model status...")
        flag = False
        for node_name, node in self.model_nodes.items():
            new_status = await self.check_node_status(node['host'], node['port'])
            if new_status != node['status']:
                flag = True
                node['status'] = new_status
                print(f"Model node {node_name} status changed to {new_status}")

                # Update the available nodes list
                for status in self.model_status:
                    if status['name'] == node['name']:
                        status['status'] = new_status
                        break

        return flag

    async def broadcast_model_status(self):
        for ws in self.model_status_clients.values():
            await ws.send_json(self.model_status)

    async def model_status_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id = id(ws)
        self.model_status_clients[client_id] = ws

        try:
            # Check the status of the model nodes
            status_changed = await self.check_model_status(request)

            # Send model status
            if status_changed:
                await self.broadcast_model_status()
            else:
                await ws.send_json(self.model_status)
                print("Model status sent: ", self.model_status)

            # Keep the connection open to send updates if needed
            async for msg in ws:
                # Handle incoming messages if necessary
                pass
        except aiohttp.web_ws.WebSocketError as e:
            print(f"WebSocket connection closed: {e}")
        finally:
            del self.model_status_clients[client_id]
            await ws.close()

        return ws
    
    async def emit_close_signal(self) -> None:
        if self.running_status:
            self.running_status = False
            await self.close()

    ## WebRTC ##
    async def index(self, request):
        content = open('index.html', 'r').read()
        return web.Response(content_type='text/html', text=content)
    
    async def websocket_handler(self, request):
        print("WebSocket connection established.")
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        room_id = request.query.get('roomId')
        if not room_id:
            await ws.close()
            return ws
        
        node = self.model_nodes.get(room_id)
        if not node:
            await ws.send_json({
                'type': 'error',
                'message': f"Model node {room_id} not found."
            })
            await ws.close()
            return ws
        
        if node['cur_conn'] >= node['max_conn']:
            await ws.send_json({
                'type': 'error',
                'message': f"Model node {room_id} is full."
            })
            await ws.close()
            return ws
        
        # Check node status
        new_status = await self.check_node_status(node['host'], node['port'])
        if new_status != node['status']:
            node['status'] = new_status
            print(f"Model node {node['name']} status changed to {new_status}")

            # Update the available nodes list
            for status in self.model_status:
                if status['name'] == node['name']:
                    status['status'] = new_status
                    break
            
            await self.broadcast_model_status()

        if node['status'] == "off":
            await ws.send_json({
                'type': 'error',
                'message': f"Model node {room_id} is not available now."
            })
            await ws.close()
            return ws

        if room_id not in self.rooms:
            assert node['cur_conn'] == 0
            self.rooms[room_id] = set()
        self.rooms[room_id].add(ws)
        print(f"Room {room_id} has {len(self.rooms[room_id])} clients.")

        node['cur_conn'] += 1
        assert node['cur_conn'] == len(self.rooms[room_id])

        # Check if the room is full
        if node['cur_conn'] >= node['max_conn']:
            for status in self.model_status:
                if status['name'] == node['name']:
                    status['full'] = True
                    break
            await self.broadcast_model_status()
        
        # Create a speechchatgpt client
        speechchatgpt_client = SpeechChatGPTClient(node['host'], node['port'], ws)
        await speechchatgpt_client.connect_to_client()

        pcs = self.pcs  # Reference to the set of peer connections

        async def leave_room():
            print("Leaving room...")
            # Decrement current connections
            node['cur_conn'] -= 1

            # Update model status if needed
            if node['cur_conn'] < node['max_conn']:
                for status in self.model_status:
                    if status['name'] == node['name']:
                        status['full'] = False
                        break
                await self.broadcast_model_status()
                
            # Clean up
            await speechchatgpt_client.close()
            self.rooms[room_id].discard(ws)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            print(f"WebSocket {ws} closed.")


        async def input_data_handler():
            try:
                async for msg in ws:
                    if msg.type == web.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        # Handle 'join_room' message
                        
                        if data['type'] == 'join_room':
                            pass
                            # Notify other clients in the room
                            # for client_ws in self.rooms[room_id]:
                            #     if client_ws != ws:
                            #         await client_ws.send_json({
                            #             'type': 'user_joined',
                            #             'name': data['name']
                            #         })
                        elif data['type'] == 'leave_room':
                            pass
                            # print("Leaving room...")
                            # await leave_room()

                        elif data['type'] == 'webrtc_signal':
                            # Handle signaling data from the client
                            signal_data = data['data']

                            if not hasattr(ws, 'pc'):
                                # Create a new RTCPeerConnection
                                pc = RTCPeerConnection()
                                pcs.add(pc)
                                ws.pc = pc  # Attach pc to the WebSocket for reference

                                @pc.on('iceconnectionstatechange')
                                async def on_iceconnectionstatechange():
                                    print(f"ICE connection state is {pc.iceConnectionState}")
                                    if pc.iceConnectionState == 'failed':
                                        await pc.close()
                                        pcs.discard(pc)

                                @pc.on('track')
                                def on_track(track):
                                    print(f"Received track: {track.kind}")
                                    if track.kind == 'audio':
                                        # Handle the audio track
                                        local_track = speechchatgpt_client.add_audio_track(track)
                                        pc.addTrack(local_track)

                            # Signal the peer connection
                            await ws.pc.setRemoteDescription(
                                RTCSessionDescription(sdp=signal_data['sdp'], type=signal_data['type'])
                            )
                            if signal_data['type'] == 'offer':
                                answer = await ws.pc.createAnswer()
                                await ws.pc.setLocalDescription(answer)
                                # Send the answer back to the client
                                await ws.send_json({
                                    'type': 'webrtc_signal',
                                    'data': {
                                        'sdp': ws.pc.localDescription.sdp,
                                        'type': ws.pc.localDescription.type
                                    }
                                })

                        elif data['type'] in ['user_text', 'system_prompt']:
                            # Handle incoming text message
                            await speechchatgpt_client.send_data(data)
                        elif data['type'] == 'open_mic':
                            speechchatgpt_client.mic_on = True
                        elif data['type'] == 'close_mic':
                            speechchatgpt_client.mic_on = False
                            # Clear the queues
                            speechchatgpt_client.audio_mind_handler.queue.queue.clear()
                            speechchatgpt_client.audio_mind_handler.frame_data_queue.queue.clear()
                            print("mic closed")
                        else:
                            print(f"Unknown message type: {data['type']}")
                    else:
                        print(f"Unhandled message type: {msg.type}")
            except Exception as e:
                print(f"Error in input_data_handler: {e}")
                traceback.print_exc()
            finally:
                print("input_data_handler terminated.")

        async def output_data_handler():
            try:
                await speechchatgpt_client.listen_for_server_data()
            except Exception as e:
                print(f"Error in output_data_handler: {e}")
                traceback.print_exc()
            finally:
                print("output_data_handler terminated.")

        async def audio_mind_handler():
            try:
                await speechchatgpt_client.audio_mind_handler.run()
            except Exception as e:
                print(f"Error in audio_mind_handler: {e}")
            finally:
                print("audio_mind_handler terminated.")

        try:
            # ... existing code ...
            tasks = [
                asyncio.create_task(input_data_handler()),
                asyncio.create_task(output_data_handler()),
                asyncio.create_task(audio_mind_handler())
            ]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            print("Cancelling pending tasks...")
            for task in pending:
                task.cancel()
            print("All tasks completed.")
        except Exception as e:
            print(f"Exception in websocket_handler: {e}")
        finally:
            await leave_room()
            # del ws  # Remove reference to ws to help garbage collection

        return ws

    async def handle_evaluation_submission(self, data, sample_id):
        try:
            submission = {
                'sample_id': sample_id,
                'client_id': data.get('client_id', 'anonymous'),
                'intelligibility': data.get('intelligibility'),
                'naturalness': data.get('naturalness'),
                'overall': data.get('overall'),
                'timestamp': data.get('timestamp')
            }

            submission_file = '/home/enpei/EnPei/GPT4o/SpeechChatGPTStreaming/_data/evaluation_submissions.jsonl'
            async with aiofiles.open(submission_file, 'a') as f:
                await f.write(json.dumps(submission) + '\n')

            print(f"Saved evaluation submission: {submission}")
        except Exception as e:
            print(f"Error saving evaluation submission: {e}")

    async def track2_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id = id(ws)
        print(f"New evaluation WebSocket connection: {client_id}")

        prompt_path = "/home/enpei/EnPei/GPT4o/SpokenLMConversation/track2_chinese_prompt.json"

        async def send_sample_audio():
            # sample from 0001 to 0010
            sample_id = np.random.randint(1, 11)
            # transform to 4 digits
            sample_id_str = f"{sample_id:04d}"
            host = request.host  # e.g., 'localhost:8080'
            scheme = 'https' if request.secure else 'http'
            sample_path_A = f'{scheme}://{host}/demo/chinese_{sample_id_str}_A.wav'
            sample_path_B = f'{scheme}://{host}/demo/chinese_{sample_id_str}_B.wav'

            # Read json
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
            
            system_prompt_A = prompts[sample_id-1]['system_prompt_A']
            system_prompt_B = prompts[sample_id-1]['system_prompt_B']

            # Send the sample audio paths to the client

            await ws.send_json({
                'type': 'sample_audio',
                'sample_id': sample_id,
                'sample_path_A': sample_path_A,
                'sample_path_B': sample_path_B,
                'system_prompt_A': system_prompt_A,
                'system_prompt_B': system_prompt_B,
            })

            return sample_id

        current_sample_id = await send_sample_audio()

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get('type') == 'evaluation_submission':
                        await self.handle_evaluation_submission(data, current_sample_id)
                    else:
                        print(f"Unknown message type: {data.get('type')}")
                elif msg.type == web.WSMsgType.ERROR:
                    print(f"WebSocket connection closed with exception: {ws.exception()}")
        except Exception as e:
            print(f"Error in evaluation_websocket: {e}")
            traceback.print_exc()
        finally:
            print(f"Evaluation WebSocket {client_id} closed")
            await ws.close()

        return ws

    def run_webrtc_server(self):
        """ Start the WebRTC connection. """

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('certificate/cert1.pem', 'certificate/privkey1.pem')

        self.app = web.Application()

        # Add middleware to set CORS headers
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            # Set CORS headers for all responses
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
            return response
        
        self.app.middlewares.append(cors_middleware)


        self.app.add_routes([
            web.get('/', self.index),
            # web.post('/offer', offer),
            web.get('/ws', self.websocket_handler),
            web.get('/ws/model-status', self.model_status_websocket),
            web.get('/ws/track2', self.track2_websocket),
        ])

        # Add static route for audio files
        self.app.router.add_static('/demo/', path='/home/enpei/EnPei/GPT4o/sarena/src/routes/challenges/track2/demo', name='demo')

        # self.app.on_startup.append(lambda app: asyncio.ensure_future(self.run()))

        # Add the background tasks to the app
        # self.app.on_startup.append(self.background_tasks)

        print("Starting WebRTC server on port 8080...")
        try:
            web.run_app(self.app, port=8080, ssl_context=ssl_context)
        except Exception as e:
            traceback.print_exc()
        finally:
            print("Web server closed")
            asyncio.run(self.emit_close_signal())

    async def close(self):
        # Close all peer connections
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        print(f"Closed {len(self.pcs)} peer connections.")

        # Close model nodes
        # for model_node in self.model_nodes:
        #     await model_node.close()
        print(f"Closed {len(self.model_nodes)} model nodes.")

        self.pcs.clear()
        # self.model_nodes.clear()

        


        