import socket
import queue
from threading import Thread
import time
from transformers import AutoTokenizer

from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.hlapi import SamplingParams

from server.common.template import DefaultProcessor, DefaultServerProcessor
from .processor_native_pytorch import LMState


class StreamSpokenLlama(object):
    input_queue: queue.Queue
    output_queue: queue.Queue
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self._build_model(config)

        self.id2state = {"default": LMState("default", self)}

        self._warmup()

    def get_system_prompt(self):
        assert "model_name" in self.config
        if self.config["model_name"] in ["taipei1/speechllama3.1-v0"]:
            return [{"role": "system", "content": """You are a chatbot, only response precisely. Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book."""}]
        elif self.config["model_name"] in ["taipei1/speechllama3.1-sft-with-back"]:
            return [{"role": "system", "content": """你是一個能和人類溝通與交流的聊天機器人。今天你正在和使用者進行一段有趣的對談 Modality: {{User: speech, Machine: speech}}."""}]
        else:
            raise NotImplementedError
        
    def _build_model(self, config: dict) -> None:
        # Initialize the model and tokenizer
        model_name = config["model_name"]
        model_path = config["model_path"]
        if model_path == "":
            model_path = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

        self.llm = GenerationExecutor.create(config["tensorrt_engine_dir"])
        print("Model loaded")

    def _warmup(self):
        print("warmup...")
        input_str = "What should you say when someone gives you a gift? You should say:"
        _ = self.llm.generate(
            self.tokenizer.encode(input_str),
            sampling_params=SamplingParams(max_tokens=10)
        )

    # output parsing
    def _is_speech_unit(self, token_id) -> bool:
        return 128256 <= token_id < 130256
    
    def _post_process(self, result_str: str):
        kms = []
        words = []
        while result_str:
            if result_str[:2] == '<|':
                idx = result_str.find('|>')
                kms.append(result_str[2:idx])
                result_str = result_str[idx+2:]
            else:
                idx = result_str.find('<|')
                if idx == -1:
                    idx = len(result_str)
                words.append(result_str[:idx])
                result_str = result_str[idx:]
        return kms, words
    
    def _parse_generated_ids(self):
        lm_state = self.id2state["default"]
        ids = []
        ed = 0
        for i, id in enumerate(lm_state.generated_ids):
            eos = (id == self.tokenizer.eos_token_id)
            if self._is_speech_unit(id) or eos:
                ids += lm_state.generated_ids[ed:i+1]
                ed = i + 1
                if eos:
                    break
        lm_state.generated_ids = lm_state.generated_ids[ed:]
        lm_state.gen_count += ed
        try:
            assert lm_state.gen_count < 1000, "too long, stop for safety"
        except:
            ids.append(self.tokenizer.eos_token_id)  # force eos

        # decode
        token = (
            self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
        )
        print("Decoded: ", token)

        input_timestamp = lm_state.last_trigger_timestamp
        kms, words = self._post_process(token)
        # print("Processed: ", kms, words)
        if words:
            res_text = {
                "type": "system_text",
                "data": "".join(words),
                "eos": False,
                "input_timestamp": input_timestamp,
            }
            print(f"Recv from Spoken Llama: {res_text}")
            self.output_queue.put(res_text)
        if kms:
            res_token = {
                "type": "system_token",
                "data": kms,
                "eos": False,
                "input_timestamp": input_timestamp,
            }
            print(f"Recv from Spoken Llama (#token): {len(kms)}")
            self.output_queue.put(res_token)

        # log
        n_tokens = lm_state.gen_count
        print(f"Forward: {time.time()-self.st:.2f}s.")
        tps = n_tokens / (time.time()-self.st)
        print(f"Rate: {int(tps)} tps ({n_tokens} tokens).")
        
        # end of stream
        if ids and ids[-1] == self.tokenizer.eos_token_id:
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

            lm_state.generated_ids.clear()
            lm_state.full_generated_ids.clear()
            lm_state.gen_count = 0
            lm_state.is_infer = False
            lm_state.responsed = True

    def _generate_n_steps(self, input_ids: list[int], gen_len: int=10):
        output = self.llm.generate(
            input_ids,
            sampling_params=SamplingParams(max_tokens=gen_len, top_k=30)
        )
        return output.outputs[0].token_ids
    
    # handler functions
    def _handle_assistant_said(self, res):
        lm_state = self.id2state["default"]
        if res.get("eos", False):
            # print("History: ", chat_history[1:])
            pass
        else:
            lm_state.update_history("Machine", res["data"])

    def _handle_reset(self, res):
        lm_state = self.id2state["default"]
        print("========== reset all state ==========")
        lm_state.reset()

    def _handle_eos(self, res):
        print("handle_eos")
        lm_state = self.id2state["default"]
        if not lm_state.is_infer and not lm_state.responsed:  # eot not predicted yet
            self._start_infer(lm_state)
    
    def _handle_no_op(self, max_gen_len=32):
        # TODO: iterate over all lm_states
        for id, lm_state in self.id2state.items():
            if not lm_state.is_infer and not lm_state.responsed:  # eot not predicted yet
                self._start_infer(lm_state)
            if not lm_state.is_infer:
                continue
            with_generation_prompt = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(with_generation_prompt, add_special_tokens=False)
            input_ids = input_ids + lm_state.full_generated_ids

            generated_ids = self._generate_n_steps(input_ids, gen_len=max_gen_len)
            print(generated_ids)

            # update LMState
            lm_state.generated_ids += generated_ids
            lm_state.full_generated_ids += generated_ids
            self._parse_generated_ids()

    def _handle_user_text_and_system_prompt(self, res):
        print("handle_user_text_and_system_prompt")
        print(res)
        lm_state = self.id2state["default"]
        if res["type"] == "user_text":
            lm_state.update_history("User", res["data"])
        elif res["type"] == "system_prompt":
            lm_state.update_history("system", res["data"]+" Modality: {{User: speech, Machine: speech}}. Speech Style: Audio Book.")

        # set state
        lm_state.is_infer = False
        lm_state.responsed = False
        lm_state.last_trigger_timestamp = res["input_timestamp"]
        lm_state.generated_ids.clear()
        lm_state.full_generated_ids.clear()
        lm_state.gen_count = 0

        # eot check
        # prompt = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=False)
        # prompt = prompt[:-10]  # remove <eot_id>
        # input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # pred_token_idx = self._generate_n_steps(input_ids, gen_len=1)[0]
        # print("EOT check: ", pred_token_idx, self.tokenizer.eos_token_id)
        # lm_state.is_infer = (pred_token_idx == self.tokenizer.eos_token_id)
        if lm_state.is_infer:
            self._start_infer(lm_state)

    def _start_infer(self, lm_state: LMState):
        print("Start infer!")
        self.st = time.time()
        lm_state.is_infer = True
    
    def run(self):
        """ loop to process input queue """
        while True:
            if self.input_queue.empty():
                res = {"type": "no_op"}
            else:
                res = self.input_queue.get()

            # update assistant history
            if res["type"] == "assistant_said":  # update assistant history
                self._handle_assistant_said(res)
            elif res["type"] == "reset":  # reset signal
                self._handle_reset(res)
            elif res.get("eos", False):  # turn take when user stops a while
                self._handle_eos(res)
            elif res["type"] == "no_op":
                self._handle_no_op()
            elif res["type"] == "user_text" or res["type"] == "system_prompt":
                self._handle_user_text_and_system_prompt(res)
            else:
                raise NotImplementedError


class SpeechLMServerProcessor(DefaultServerProcessor):
    def __init__(self, config):
        super().__init__(config)
    
    def _setup_model(self):
        print("Run Stream Spoken Llama on new thread!")
        self.model = StreamSpokenLlama(self.config["model"])  # all processors will share one model
        Thread(target=self.model.run, daemon=True).start()

    def _create_processor(self, client_socket: socket.socket, addr) -> DefaultProcessor:
        p = DefaultProcessor(self.config, client_socket, addr)
        p.connect_model(self.model)
        return p
