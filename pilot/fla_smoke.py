import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MID="Qwen/Qwen3.5-2B"
try:
 m=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float16,device_map={"":0})
except Exception:
 from transformers import AutoModelForImageTextToText
 m=AutoModelForImageTextToText.from_pretrained(MID,dtype=torch.float16,device_map={"":0})
t=AutoTokenizer.from_pretrained(MID)
x=t("Hello, my name is",return_tensors="pt").to(0)
y=m.generate(**x,max_new_tokens=8,do_sample=False)
assert torch.isfinite(m(**x).logits).all()
print("SMOKE_OK",t.decode(y[0]))
