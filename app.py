import sys
import socket
import sounddevice as sd
import numpy as np
import threading
import asyncio
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QScrollArea, QWidget, QLineEdit, QLabel, QMainWindow,
                             QListWidget, QListWidgetItem, QSpacerItem, QSizePolicy)

from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal
from PyQt5.QtGui import QIcon

from client._client import AsyncClient
from client.audio_stream import AudioStreamPlayer
from server.utils import handle_asyncio_exception


# SERVER_IP = '140.112.21.24'
# SERVER_PORT = 16006
SERVER_IP = 'localhost'
SERVER_PORT = 43007

    
class MessageWidget(QWidget):
    """
    User Text: Blue background, right-aligned.
    User Audio: Pink background, right-aligned.
    System Response: Grey background, left-aligned.
    """
    def __init__(self, text, message_type):
        super().__init__()
        self.text = text
        self.message_type = message_type
        self.label = None
        self.initUI()

    def initUI(self):
        if self.layout() is None:  # Check if the widget already has a layout
            layout = QHBoxLayout()
            self.setLayout(layout)
        else:
            layout = self.layout()

        if self.label is None:  # Create the QLabel if it doesn't exist
            self.label = QLabel(self.text)
            self.label.setWordWrap(True)
            self.label.setMaximumWidth(400)  # Maximum width for message bubbles

            if self.message_type == 'user_text_typed':
                self.label.setStyleSheet("QLabel { background-color: #add8e6; color: black; border-radius: 10px; padding: 8px; }")
                spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
                layout.addSpacerItem(spacer)
                layout.addWidget(self.label)
            elif self.message_type == 'user_text':
                self.label.setStyleSheet("QLabel { background-color: #ffb6c1; color: black; border-radius: 10px; padding: 8px; }")
                spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
                layout.addSpacerItem(spacer)
                layout.addWidget(self.label)
            elif self.message_type == 'system_text':
                self.label.setStyleSheet("QLabel { background-color: #d3d3d3; color: black; border-radius: 10px; padding: 8px; }")
                layout.addWidget(self.label)
                spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
                layout.addSpacerItem(spacer)
        else:
            # Update existing QLabel
            self.label.setText(self.text)
            # No need to reset the stylesheet unless it changes dynamically based on some condition


class AudioStreamer(QWidget):
    update_message_signal = pyqtSignal(object, str)  # Updated to pass message type as well

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._loop  = loop  # asyncio loop
        self.initUI()
        self.is_recording = False
        self.stream = None
        self.sock = None
        self.update_message_signal.connect(self.display_message)
        self.last_server_msg_widget = None  # Keep track of the last server message widget
        self.last_openai_msg_widget = None  # Keep track of the last OpenAI message widget
        self.system_text_showing = False  # Keep track of the text shown
        self.system_audio_playing = False  # Keep track of the audiostream played

        # client wrapper
        self.client = AsyncClient(SERVER_IP, SERVER_PORT)
        future = asyncio.run_coroutine_threadsafe(self.client.run(), self._loop)
        _ = future.result()  # wait until client.run finished

        # audio player thread
        self.audio_stream_player = AudioStreamPlayer()
        self.audio_stream_player.start_stream()

        self.listen_for_server_data()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.inputBox = QLineEdit(self)
        self.inputBox.returnPressed.connect(self.sendUserMessage)
        self.micButton = QPushButton(QIcon("microphone_icon.png"), "", self)
        self.micButton.clicked.connect(self.toggle_recording)
        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.micButton)
        bottomLayout.addWidget(self.inputBox)
        layout.addWidget(self.scrollArea)
        layout.addLayout(bottomLayout)
        self.setLayout(layout)
        self.setWindowTitle('SPML GPT')
        self.setGeometry(300, 300, 600, 600)

    # audio input
    def toggle_recording(self):
        if not self.is_recording:
            self.start_streaming()
        else:
            self.stop_streaming()
            # self.update_message_signal.emit("Audio recording stopped.", "user_audio")

    def start_streaming(self):
        self.is_recording = True
        self.micButton.setIcon(QIcon("mic_on_icon.png"))
        self.stream = sd.InputStream(
            device=1,
            samplerate=16000,
            channels=1,
            dtype='int16',
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_streaming(self):
        self.is_recording = False
        self.micButton.setIcon(QIcon("microphone_icon.png"))
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def audio_callback(self, indata: np.ndarray, frames, _time, status):
        # submit to asyncio event loop
        fut = asyncio.run_coroutine_threadsafe(self.client.send_audio(indata), self._loop)
        fut.add_done_callback(handle_asyncio_exception)

    # text input
    def sendUserMessage(self):
        message = self.inputBox.text()
        self.inputBox.clear()
        if message:
            self.display_message(message, "user_text_typed")
            self.send_text_request(message)
    
    def send_text_request(self, message: str):
        # submit to asyncio event loop
        fut = asyncio.run_coroutine_threadsafe(self.client.send_user_text(message), self._loop)
        fut.add_done_callback(handle_asyncio_exception)

    def listen_for_server_data(self):
        # submit to asyncio event loop
        async def f():
            try:
                stream = self.client.recv_stream()
                async for res in stream:
                    if res["type"] == "system_audio":
                        self.receive_system_audio(res)
                    elif res["type"] == "user_text":
                        self.receive_user_text(res)
                    elif res["type"] == "system_text":
                        self.receive_system_text(res)
                    elif res["type"] == "command":
                        print(res["data"])
                    else:
                        raise NotImplementedError
            except (ConnectionResetError, BrokenPipeError):
                print("Client connection closed")
                raise
            except Exception as e:
                raise
                
        fut = asyncio.run_coroutine_threadsafe(f(), self._loop)
        fut.add_done_callback(self.handle_server_close)

    def handle_server_close(self, fut: asyncio.Task):
        if fut.exception():
            try:
                # re-raise exception
                fut.result()
            except Exception as e:
                traceback.print_exc()
        self.close()
    
    # handle output
    def receive_system_audio(self, res):
        # print(res)
        if res.get("eos", False):
            print("system audio break")
            self.system_audio_playing = False
            return
        if not self.system_audio_playing:
            if max(res["data"]) > 0:  # first non-empty chunk
                print(f"ASR + LLM + TTS latency: {time.time()-res['input_timestamp']:.2f}s.")
                self.system_audio_playing = True
        self.audio_stream_player.put(res["data"])
    
    def receive_user_text(self, res):
        # print(res)
        if res.get("eos", False):
            print("user break")
            self.update_message_signal.emit(None, "user_text")  # thread safe
            return
        print(f"ASR latency: {time.time()-res['input_timestamp']:.2f}s.")
        self.update_message_signal.emit(res["data"], "user_text")

    def receive_system_text(self, res):
        # print(res)
        if res.get("eos", False):
            print("system break")
            self.update_message_signal.emit(None, "system_text")  # thread safe
            self.system_text_showing = False
            return
        if not self.system_text_showing:
            # print(f"LLM latency: {time.time()-res['turn_take_timestamp']:.2f}s.")
            print(f"ASR + LLM latency: {time.time()-res['input_timestamp']:.2f}s.")
            self.system_text_showing = True
        self.update_message_signal.emit(res["data"], "system_text")

    # @pyqtSlot(str, str)
    def display_message(self, message, message_type):
        if message_type == "user_text_typed":
            msg_widget = MessageWidget(message, message_type)
            self.scrollAreaLayout.addWidget(msg_widget)
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
            self.last_server_msg_widget, self.last_openai_msg_widget = None, None
        
        elif message_type == "user_text":
            if message is None:
                self.last_server_msg_widget = None
                return
            if self.last_server_msg_widget is None:
                msg_widget = MessageWidget(message, message_type)
                self.scrollAreaLayout.addWidget(msg_widget)
                self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
                self.last_server_msg_widget, self.last_openai_msg_widget = msg_widget, None
            else:
                # Update the last system message text
                self.last_server_msg_widget.text += message
                self.last_server_msg_widget.initUI()  # Refresh the UI with updated text
                
        elif message_type == "system_text":
            if message is None:
                self.last_openai_msg_widget = None
                return
            if self.last_openai_msg_widget is None:
                msg_widget = MessageWidget(message, message_type)
                self.scrollAreaLayout.addWidget(msg_widget)
                self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
                self.last_server_msg_widget, self.last_openai_msg_widget = None, msg_widget
            else:
                # Update the last system message text
                self.last_openai_msg_widget.text += message
                self.last_openai_msg_widget.initUI()  # Refresh the UI with updated text
        else:
            raise NotImplementedError


class AsyncIOThread(threading.Thread):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._loop = loop
        self.daemon = True

    def run(self) -> None:
        self._loop.run_forever()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    loop = asyncio.new_event_loop()
    thd = AsyncIOThread(loop)
    thd.start()

    # QT event loop
    ex = AudioStreamer(loop)
    ex.show()
    sys.exit(app.exec_())
