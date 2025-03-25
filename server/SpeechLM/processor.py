# Deprecated
import os
import socket
import asyncio
import pickle
import torch
import queue
from threading import Thread
import time
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from ..utils import length_prefixing, recv_with_length_prefixing, handle_asyncio_exception


class ForwardCriterion(object):
    def __init__(self):
        self.current_trigger_timestamp = None
        self.last_trigger_timestamp = None
    
    def exec(self, info: dict):
        res = info["res"]
        is_blank = res["data"] is None or res.get("eos", False)
        is_repeat = self.last_trigger_timestamp is not None and self.current_trigger_timestamp == self.last_trigger_timestamp
        return is_blank and not is_repeat and self.current_trigger_timestamp is not None
    
    def reset(self):
        self.current_trigger_timestamp = None
        self.last_trigger_timestamp = None


class ForwardCriterion2(object):
    def __init__(self):
        self.current_trigger_timestamp = None
        self.last_trigger_timestamp = None
        self.eot_threshold = 0.00001
    
    def set_eot_threshold(self, x:float) -> None:
        self.eot_threshold = x
    
    def exec(self, info: dict):
        res, eot_prob = info["res"], info["eot_prob"]
        is_eos = res.get("eos", False)
        is_eot_pred = eot_prob > self.eot_threshold
        is_blank = res["data"] is None or res.get("eos", False)
        is_repeat = self.last_trigger_timestamp is not None and self.current_trigger_timestamp == self.last_trigger_timestamp
        return (is_eos or is_eot_pred) and not is_repeat and self.current_trigger_timestamp is not None
    
    def reset(self):
        self.current_trigger_timestamp = None
        self.last_trigger_timestamp = None


class StreamSpokenLlama(object):
    input_queue: queue.Queue
    output_queue: queue.Queue
    
    SYSTEM_PROMPT = [{"role": "system", "content": """You are a chatbot, only response precisely. Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."""}]

    def __init__(self, model_path) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self._current_task = None
        self._stop_signal = False
        self._build_model(model_path)
        self._unit_signal = False
    
    def _build_model(self, model_path) -> None:
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, device_map="auto",
            quantization_config={"load_in_4bit": True},
            low_cpu_mem_usage=True
        )
        print("Model loaded")
    
    def _calc_eot_prob(self, chat_history: dict) -> float:
        prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
        # print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.llm.device)
        input_ids = input_ids[:, :-1]
        logits = self.llm(input_ids).logits
        eot_prob = torch.softmax(logits, dim=-1)[0, -1, 128009]
        print(f"EOT prob: ", eot_prob.item())
        return eot_prob.item()

    def _update_history(self, chat_history: list, role: str, new_content: str):
        if chat_history[-1]["role"] == role:
            msg = chat_history.pop()
            if role == "system":
                msg = {"role": msg["role"], "content": new_content}
            else:
                msg = {"role": msg["role"], "content": msg["content"] + new_content}
            chat_history.append(msg)
        else: 
            chat_history.append({"role": role, "content": new_content})

    def _post_process(self, result_str: str):
        results = result_str.replace('|>', '|> ').replace('<|', ' <|').replace('  ', ' ').split(' ')
        kms = []
        words = []
        for u in results:
            if u == "<|eot_id|>":
                break
            if u[:2] == '<|' and u[-2:] == '|>':
                kms.append(u[2:-2])
            else:
                words.append(u)
        return kms, words

    def _stop_current_task(self):
        self._stop_signal = True  # stop current_task gracefully by specific signal
        self._current_task.join()
        self._stop_signal = False

    def _async_forward(self, x: dict):
        chat_history, input_timestamp = x["chat_history"], x["input_timestamp"]
        prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        
        # Generate text in a separate thread
        generation_kwargs = dict(inputs, max_new_tokens=2048, streamer=streamer)
        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()  # we can't really suspend this thread unless we customized generate()
        
        # Iterate over the generated tokens
        eot_token = "<|eot_id|>"
        for token in streamer:
            if self._stop_signal:
                break
            if not token.strip():  # empty
                continue

            # Parsing
            kms, words = self._post_process(token)
            if kms or words:
                res_text = {
                    "type": "system_text",
                    "data": " ".join(words),
                    "eos": False,
                    "input_timestamp": input_timestamp,
                }
                res_token = {
                    "type": "system_token",
                    "data": kms,
                    "eos": False,
                    "input_timestamp": input_timestamp,
                }
                self.n_tokens += len(kms) + len(words)
                print(f"Recv from Spoken Llama: {res_text}")
                print(f"Recv from Spoken Llama (#token): {len(kms)}")
                self.output_queue.put(res_text)
                self.output_queue.put(res_token)
                
            # end of stream
            if eot_token in token:
                res_text = {
                    "type": "system_text",
                    "data": None,
                    "eos": True,
                    "input_timestamp": input_timestamp,
                }
                res_token = {
                    "type": "system_token",
                    "data": None,
                    "eos": True,
                    "input_timestamp": input_timestamp,
                }
                self.output_queue.put(res_text)
                self.output_queue.put(res_token)
                break
        thread.join()  # blocks until generate() done

    def forward(self, x: dict):
        """ forward a single chunk of data """
        st = time.time()
        self.n_tokens = 0
        print(f"Forward: {x}")
        if self._current_task is not None and self._current_task.is_alive():
            # self._stop_current_task()
            pass
        self._current_task = Thread(target=self._async_forward, args=(x,))
        self._current_task.start()
        self._current_task.join()  # TODO: fake async since buggy
        print(f"Forward: {time.time()-st:.2f}s.")
        tps = self.n_tokens / (time.time()-st)
        print(f"Rate: {int(tps)} tps ({self.n_tokens} tokens).")

    def run(self):
        """ loop to process input queue """
        chat_history = copy.deepcopy(StreamSpokenLlama.SYSTEM_PROMPT)
        forward_criterion = ForwardCriterion2()
        while True:
            if self.input_queue.empty():  # define blank format
                res = {
                    "type": "user_text",
                    "data": None,
                    "eos": False,
                    "input_timestamp": None
                }
            else:
                res = self.input_queue.get()

            # update assistant history
            if res["type"] == "assistant_said":
                if res.get("eos", False):
                    # print("History: ", chat_history[1:])
                    pass
                else:
                    self._update_history(chat_history, "Machine", res["data"])
                continue

            # reset signal
            if res["type"] == "reset":
                print("========== reset all state ==========")
                chat_history = copy.deepcopy(StreamSpokenLlama.SYSTEM_PROMPT)
                forward_criterion.reset()
                continue
            
            # blank or user input
            assert res["type"] == "user_text" or res["type"] == "system_prompt"      
            info = {"res": res}
            if isinstance(forward_criterion, ForwardCriterion2):
                info["eot_prob"] = 0.0
            if res["data"] is not None:  # update user text if not blank (eos is considered blank)
                if res["type"] == "system_prompt":
                    res["data"] += " Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."
                    print("========== system prompt ==========")
                    print("SYSTEM PROMPT: ", res["data"])
                    self._update_history(chat_history, "system", res["data"])
                elif res["type"] == "user_text":
                    self._update_history(chat_history, "User", res["data"])
                forward_criterion.current_trigger_timestamp = res["input_timestamp"]
                if isinstance(forward_criterion, ForwardCriterion2):
                    info["eot_prob"] = self._calc_eot_prob(chat_history)  # currently too slow
                    # info["eot_prob"] = 0
            
            if forward_criterion.exec(info):
                forward_criterion.last_trigger_timestamp = forward_criterion.current_trigger_timestamp
                self.forward({
                    "chat_history": copy.deepcopy(chat_history),
                    "input_timestamp": forward_criterion.current_trigger_timestamp,
                })
            
            # handle user break after forward detection
            eos = res.get("eos", False)
            if eos:
                # print("History: ", chat_history[1:])
                forward_criterion.reset()


class Processor(object):
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
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            print(f"Connection from {self.addr} closed.")
            self.client_socket.close()
            self.client_socket_status = False
            self.emit_reset_signal()

    async def output_data(self):
        loop = asyncio.get_event_loop()
        try:
            while self.client_socket_status:
                try:
                    res = self.model.output_queue.get_nowait()
                    await loop.sock_sendall(self.client_socket, length_prefixing(res))
                except queue.Empty:
                    await asyncio.sleep(0.01)  # return control to event loop
        except:
            pass

    async def process(self):
        assert self.model is not None, "Please connect the model by calling connect_model(model) first"
        self.client_socket_status = True
        await asyncio.gather(self.input_data(), self.output_data())
        print("Connection gracefully shutdown.")


class LlamaServerProcessor(object):
    """ This is only a wrapper class. """
    def __init__(self, config):
        self.config = config
        # model thread, all processor will share one model
        self.model = StreamSpokenLlama(config['model_path'])
        Thread(target=self.model.run, daemon=True).start()
        print("Run Stream Spoken Llama on new thread!")
    
    async def process(self, client_socket: socket.socket, addr):  # handle one client connection
        loop = asyncio.get_event_loop()
        processor = Processor(self.config, client_socket, addr)
        processor.connect_model(self.model)

        # main process loop
        fut = loop.create_task(processor.process())
        fut.add_done_callback(handle_asyncio_exception)
