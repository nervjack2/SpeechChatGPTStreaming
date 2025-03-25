import socket


class IProcessor(object):
    """ Processor interface for single client """
    def __init__(self, config: dict, client_socket: socket.socket, addr) -> None:
        raise NotImplementedError

    async def exec(self) -> None:
        """ main loop """
        raise NotImplementedError


class IServerProcessor(object):
    def __init__(self, config: dict) -> None:
        raise NotImplementedError

    async def process(self, client_socket: socket.socket, addr) -> None:
        """ handle connection from single client """
        raise NotImplementedError
