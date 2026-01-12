import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class SummarizationHandler(BaseHandler):
    def initialize(self, context):
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = Path(context.system_properties.get("model_dir"))
        self.model_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        state_dict_path = model_dir / "model.pt"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.sys_prompt = self._load_prompt("samsum_sysprompt.txt")
        self.usr_prompt = self._load_prompt("samsum_usrprompt.txt")

        config_path = model_dir / "model_config.json"
        self.default_generation_config = {}
        if config_path.exists():
            self.default_generation_config = json.loads(config_path.read_text())

    def _load_prompt(self, filename: str) -> str:
        prompt_path = self.model_dir / filename
        if not prompt_path.exists():
            return "You are a dialogue summarizer."
        return prompt_path.read_text().strip()

    def preprocess(self, data):
        requests = []
        for item in data:
            payload = item.get("data") or item.get("body") or item
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)

            requests.append(payload)
        return requests

    def inference(self, requests):
        responses = []
        for payload in requests:
            dialogue = payload.get("dialogue") or payload.get("inputs")
            if dialogue is None:
                raise ValueError("Missing 'dialogue' or 'inputs' in request payload")

            generation_config = dict(self.default_generation_config)
            for key in ("max_new_tokens", "temperature", "top_p", "top_k", "do_sample"):
                if key in payload:
                    generation_config[key] = payload[key]

            user_prompt = self.usr_prompt.format(dialogue=dialogue)
            prompt_messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, **generation_config)

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            responses.append({"summary": content})
        return responses

    def postprocess(self, inference_output):
        return inference_output


_service = SummarizationHandler()
_initialized = False


def handle(data, context):
    if _service is None:
        return None
    global _initialized
    if not _initialized:
        _service.initialize(context)
        _initialized = True
    if data is None:
        return None
    return _service.handle(data, context)
