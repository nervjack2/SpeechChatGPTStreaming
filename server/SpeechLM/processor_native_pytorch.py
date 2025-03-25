import socket
import torch
import queue
from threading import Thread
import time
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from server.common.template import DefaultProcessor, DefaultServerProcessor
from .utils import multinomial_top_k_sampling


class LMState(object):
    def __init__(self, id: str, model):
        self.id = id

        # history
        self.chat_history_init = model.get_system_prompt()
        self.chat_history = self.chat_history_init
        self.past_key_values = DynamicCache()
        self.last_token = None
        self.last_trigger_timestamp = None

        # state control
        self.is_infer = False
        self.responsed = True
        self.system_prompt_applied = False

        # generation
        self.full_generated_ids = []
        self.generated_ids = []
        self.gen_count = 0
    
    def update_history(self, role: str, new_content: str):
        if self.chat_history[-1]["role"] == role:
            msg = self.chat_history.pop()
            if role == "system":
                msg = {"role": msg["role"], "content": new_content}
            else:
                msg = {"role": msg["role"], "content": msg["content"] + new_content}
            self.chat_history.append(msg)
        else: 
            self.chat_history.append({"role": role, "content": new_content})

    def reset(self):
        # history
        self.chat_history = self.chat_history_init
        self.past_key_values = DynamicCache()
        self.last_token = None
        self.last_trigger_timestamp = None

        # state control
        self.is_infer = False
        self.responsed = True
        self.system_prompt_applied = False

        # generation
        self.generated_ids = []
        self.gen_count = 0


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_path
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, device_map="auto",
            quantization_config={"load_in_4bit": True},
            low_cpu_mem_usage=True,
            cache_dir=model_path
        )
        print("Model loaded")

    def _warmup(self):
        print("warmup...")
        input_str = "What should you say when someone gives you a gift? You should say:"
        inputs = self.tokenizer(input_str, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            input_ids = inputs.input_ids.to(self.llm.device)
            _ = self.llm(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
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
        print("Processed: ", kms, words)
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

    # forward functions
    @torch.no_grad()
    def _prefill(self, input_ids, kv_cache):
        """ forward a segment, return the next token and kv_cache """
        outputs = self.llm(
            input_ids=input_ids,
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # last token idx
        return pred_token_idx, kv_cache

    @torch.no_grad()
    def _decode_one_token(self, last_token, kv_cache):
        outputs = self.llm(
            input_ids=last_token,
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        # pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # last token idx
        # print(pred_token_idx)

        # multinomaial + top k sampling
        logits = outputs.logits[:, -1, :]
        pred_token_idx = multinomial_top_k_sampling(logits, k=30).unsqueeze(1)
        # print(pred_token_idx)

        return pred_token_idx, kv_cache
    
    @torch.no_grad()
    def _decode_n_steps(self, last_token, kv_cache, gen_len: int=10):
        print("generate tokens")
        generated_ids = []
        for _ in range(gen_len):
            generated_ids.append(last_token.item())
            last_token, kv_cache = self._decode_one_token(last_token, kv_cache)
            if last_token == self.tokenizer.eos_token_id:
                break
        return generated_ids, last_token, kv_cache

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
    
    def _handle_no_op(self, max_gen_len=10):
        # TODO: iterate over all lm_states
        for id, lm_state in self.id2state.items():
            if not lm_state.is_infer:
                continue
            generated_ids, last_token, kv_cache = self._decode_n_steps(
                lm_state.last_token, lm_state.past_key_values, max_gen_len
            )
            
            # update LMState
            lm_state.generated_ids += generated_ids
            lm_state.full_generated_ids += generated_ids
            lm_state.last_token, lm_state.past_key_values = last_token, kv_cache
            self._parse_generated_ids()

    def _handle_user_text_and_system_prompt(self, res):
        print("handle_user_text_and_system_prompt")
        lm_state = self.id2state["default"]
        chat_history_before = copy.deepcopy(lm_state.chat_history)
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

        # prepare prompt
        if not lm_state.system_prompt_applied or res["type"] == "system_prompt":  # first pass
            prompt = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=False)
            prompt = prompt[:-10]  # remove <eot_id>
            lm_state.system_prompt_applied = True
        else:
            before = self.tokenizer.apply_chat_template(chat_history_before, tokenize=False, add_generation_prompt=False)
            before = before[:-10]  # remove <eot_id>
            after = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=False)
            after = after[:-10]  # remove <eot_id>
            prompt = after[len(before):]  # get difference
        print("Prompt: ", prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.llm.device)
        pred_token_idx, past_key_values = self._prefill(input_ids, lm_state.past_key_values)
        lm_state.last_token, lm_state.past_key_values = pred_token_idx, past_key_values

        # eot check
        print("EOT check: ", pred_token_idx, self.tokenizer.eos_token_id)
        lm_state.is_infer = (pred_token_idx.item() == self.tokenizer.eos_token_id)
        if lm_state.is_infer:
            self._start_infer(lm_state)
        # # print(past_key_values.key_cache[0].shape)
        # print(past_key_values.value_cache[0].shape)

    def _start_infer(self, lm_state: LMState):
        print("Start infer!")
        self.st = time.time()
        without_generation_prompt = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=False)
        with_generation_prompt = self.tokenizer.apply_chat_template(lm_state.chat_history, tokenize=False, add_generation_prompt=True)
        diff = with_generation_prompt[len(without_generation_prompt):]
        diff = f"<|eot_id|>{diff}"
        print("Prompt: ", diff)
        inputs = self.tokenizer(diff, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.llm.device)
        pred_token_idx, past_key_values = self._prefill(input_ids, lm_state.past_key_values)
        lm_state.last_token, lm_state.past_key_values = pred_token_idx, past_key_values
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
