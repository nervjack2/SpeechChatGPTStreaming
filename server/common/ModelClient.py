import uuid
import asyncio
import traceback
import queue
import threading
import collections


class IServing(object):
    def setup_connection(self, uid):
        raise NotImplementedError
    
    def close_connection(self, uid):
        raise NotImplementedError


class AsyncModelClient(object):

    pending_requests: dict[uuid.UUID, asyncio.Future]
    
    def __init__(self, model: IServing) -> None:
        self.uid = uuid.uuid4()
        self.model = model
        print(f"[Model client({self.uid})] Setup connection.")
        self.response_queue, self.request_queue = model.setup_connection(self.uid)

        # Dictionary to map request uuid to asyncio.Future objects
        self.pending_requests = {}

        # run
        self.running_status = True
        self.fut = asyncio.create_task(self.receive_responses())
    
    async def send_request(self, req: dict):
        assert self.running_status, "Client has been closed."
        request_id = uuid.uuid4()
        future = asyncio.get_event_loop().create_future()
        
        # Save the future in the pending_requests dictionary
        self.pending_requests[request_id] = future
        
        self.request_queue.put_nowait({
            "headers": {
                "uid": self.uid,
                "req_uid": request_id,
            },
            "request_body": req,
        })
        # print(f"Request:", {
        #     "uid": self.uid,
        #     "req_uid": request_id,
        # })
        
        # Return the future, which will be completed later when the response is received
        await future
        return future.result()
    
    async def receive_responses(self):
        while True:
            try:
                response = self.response_queue.get_nowait()
                # Assume response contains 'uuid' and 'data'
                request_id = response['headers']['req_uid']
                # Complete the future with the response data
                if request_id in self.pending_requests:
                    if not self.pending_requests[request_id].done():
                        self.pending_requests[request_id].set_result(response['data'])
                    # Remove from pending requests
                    del self.pending_requests[request_id]
                    # print("Request done:", response["headers"])
            except queue.Empty:
                pass
            except Exception as e:
                traceback.print_exc()
            await asyncio.sleep(0.01)  # return control to event loop

    async def close(self):
        self.running_status = False
        self.model.close_connection(self.uid)
        try:
            self.fut.cancel()
            await self.fut
        except asyncio.CancelledError:
            pass

        # shut down all send_request()
        for request_id, future in self.pending_requests.items():
            future.cancel()
        self.pending_requests = {}


class SyncModelClient(object):
    
    pending_requests: dict[uuid.UUID, queue.Queue]
    
    def __init__(self, model: IServing) -> None:
        self.uid = uuid.uuid4()
        self.model = model
        print(f"[Model client({self.uid})] Setup connection.")
        self.response_queue, self.request_queue = model.setup_connection(self.uid)

        # Dictionary to map request uuid to asyncio.Future objects
        self.pending_requests = collections.defaultdict(queue.Queue())

        # run
        self.running_status = True
        self.thd = threading.Thread(self.receive_responses)
        self.thd.start()
    
    def send_request(self, req: dict):
        assert self.running_status, "Client has been closed."
        request_id = uuid.uuid4()
        q = self.pending_requests[request_id]
        
        self.request_queue.put({
            "headers": {
                "uid": self.uid,
                "req_uid": request_id,
            },
            "request_body": req,
        })
        # print(f"Request:", {
        #     "uid": self.uid,
        #     "req_uid": request_id,
        # })
        
        res = q.get()
        if res is None:
            return
        return res
    
    def receive_responses(self):
        while True:
            response = self.response_queue.get()
            if response is None:
                return
            request_id = response['headers']['req_uid']
            assert request_id in self.pending_requests
            self.pending_requests[request_id].put(response['data'])
            del self.pending_requests[request_id]
            # print("Request done:", response["headers"])

    def close(self):
        self.running_status = False
        self.model.close_connection(self.uid)
        self.response_queue.put(None)  # shut down thd
        self.thd.join()

        # shut down all send_request()
        for request_id, q in self.pending_requests.items():
            q.put(None)
        self.pending_requests = {}
