import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flash_pref import shared_prefix

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(
    model_id, attn_implementation="flash_attention_2", use_cache=False, torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = "What is the next 10 numbers of this sequence: " + ", ".join(str(x) for x in range(500))
chosen_response = ", ".join(str(x) for x in range(500, 500 + 10))
rejected_response = ", ".join(str(x) for x in range(500, 500 + 10, 2))

conversations = [
    [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_response}],
    [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_response}],
]
inputs = tokenizer.apply_chat_template(
    conversations, tokenize=True, padding=True, return_tensors="pt", return_dict=True
).to("cuda")

# ===== MAGIC HERE =====
with shared_prefix(model, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask):
    output = model(**inputs)
    output.logits.backward(torch.randn_like(output.logits))
