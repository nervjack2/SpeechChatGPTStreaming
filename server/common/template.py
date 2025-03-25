import asyncio
import socket
import pickle
import queue
import traceback

from server.base import IProcessor, IServerProcessor
from server.utils import recv_with_length_prefixing, length_prefixing, run_parallel_tasks


class DefaultProcessor(IProcessor):
    """ Default processor for single client. """
    def __init__(self, config, client_socket: socket.socket, addr):
        self.config = config
        self.client_socket = client_socket
        self.addr = addr

        self.model = None

    def connect_model(self, model) -> None:
        self.model = model

    def emit_reset_signal(self) -> None:
        """ reset the model state on the other thread """
        self.model.input_queue.put({"type": "reset"})
    
    async def input_data(self):
        try:
            while True:
                res = await recv_with_length_prefixing(client_socket=self.client_socket)
                if not res:
                    break
                res = pickle.loads(res)

                # special command
                if res["type"] == "reset":  # special command
                    self.emit_reset_signal()
                    continue

                # print(f"Put {res}")
                self.model.input_queue.put(res)
        except (ConnectionResetError, BrokenPipeError):
            print("Client connection closed")
            raise
        except Exception as e:
            raise
        print(f"Connection from {self.addr} closed.")

    async def output_data(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                res = self.model.output_queue.get_nowait()
                await loop.sock_sendall(self.client_socket, length_prefixing(res))
            except queue.Empty:
                await asyncio.sleep(0.01)  # return control to event loop

    async def exec(self):
        assert self.model is not None, "Please connect the model by calling connect_model(model) first"
        coros = [
            self.input_data(),
            self.output_data(),
        ]
        tasks = [asyncio.create_task(coro) for coro in coros]
        await run_parallel_tasks(tasks)

        # clean up
        self.client_socket.close()
        self.emit_reset_signal()


class DefaultServerProcessor(IServerProcessor):
    """
    Template for server processors.
    Need to implement _setup_model() and _create_processor().
    """
    def __init__(self, config):
        self.config = config
        self._setup_model()

    def _setup_model(self) -> None:
        raise NotImplementedError

    def _create_processor(self, client_socket: socket.socket, addr) -> IProcessor:
        raise NotImplementedError
    
    async def process(self, client_socket: socket.socket, addr):  # handle one client connection
        try:
            processor = self._create_processor(client_socket, addr)
            await processor.exec()
        except Exception as e:
            traceback.print_exc()
        finally:
            del processor
            print(f"Connection from {addr} gracefully shutdown.")
