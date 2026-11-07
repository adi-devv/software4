from typing import Protocol
import re


class LLM(Protocol):
    input_limit: int
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str: ...
    def count_tokens(self, text: str) -> int: ...


class FakeLLM:
    """Offline prompt reader for tests; understands simple field notes and exact examples."""
    input_limit = 1_000_000

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens=256):
        if prompt.startswith('Distill lessons:\n'):
            lines = []
            for line in prompt.splitlines():
                if line.startswith(('Fact: ', 'Correction: ')):
                    lines.append(line.split(': ', 1)[1])
            return '\n'.join(dict.fromkeys(lines))
        context, sep, question = prompt.rpartition('\n\nTask:\n')
        if not sep:
            return 'Final answer: unknown'
        q = question.lower()
        matches = []
        for line in context.splitlines():
            line = line.strip()
            fact = re.search(r'(?:the )?(.+?) of (.+?) is (?:now )?([^.,]+)', line, re.I)
            record = re.search(r'Record: (.+?), (.+?): ([^.]+)', line, re.I)
            if fact:
                attr, entity, value = fact.groups()
            elif record:
                entity, attr, value = record.groups()
            else:
                continue
            attr = re.sub(r'^.*?(?:Field note:|Update notice:)\s*(?:the )?', '', attr, flags=re.I)
            aliases = {'leader': ('leader', 'leads', 'led'), 'main export': ('export',),
                       'founding year': ('founded', 'year'), 'population': ('population', 'people'),
                       'ally': ('ally', 'allied')}
            if entity.lower() in q and any(x in q for x in aliases.get(attr.lower(), (attr.lower(),))):
                matches.append(value.strip())
        return 'Final answer: ' + (matches[-1] if matches else 'unknown')


class HFLLM:
    def __init__(self, model='Qwen/Qwen3.5-2B', device='cuda', input_limit=32768):
        self.model_name, self.device = model, device
        self.input_limit = input_limit
        self._model = self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device).eval()

    def _format(self, text):
        self._load()
        options = {'enable_thinking': False} if 'enable_thinking' in (self._tokenizer.chat_template or '') else {}
        return self._tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                   tokenize=False, add_generation_prompt=True, **options)

    def count_tokens(self, text):
        formatted = self._format(text)
        return len(self._tokenizer.encode(formatted, add_special_tokens=False))

    def generate(self, prompt, max_new_tokens=256):
        self._load()
        import torch
        inputs = self._tokenizer(self._format(prompt), return_tensors='pt', add_special_tokens=False).to(self.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
