import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT-ryan1.5b-float16', bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]',
    pad_token='[PAD]', mask_token='[MASK]'
)

model = AutoModelForCausalLM.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16', pad_token_id=tokenizer.eox_token_id,
    torch_dtype='auto', low_cpu_mem_usage=True
).to(device=device, non_blocking=True)

model.eval()

