import asyncio
import time
import traceback


class TurnTaker(object):
    def __init__(self, patience=1.0) -> None:
        self.patience = patience
        self.flag = asyncio.Event()
        self.wait_fut = None

    async def tick(self):
        await self.interrupt()
        self.wait_fut = asyncio.create_task(self._waiting())

    async def interrupt(self):
        if self.wait_fut is None or self.wait_fut.done():
            pass
        else:
            self.wait_fut.cancel()
            await self.wait_fut
        self.wait_fut = None

    async def _waiting(self):
        try:  # handle exceptions internally
            await asyncio.sleep(self.patience)
            self.flag.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exc()

    def emit_signal(self):
        """ trigger turn take immediately """
        # self.flag = True
        self.flag.set()

    async def turn_take_signal(self):
        await self.flag.wait()
        self.flag.clear()
