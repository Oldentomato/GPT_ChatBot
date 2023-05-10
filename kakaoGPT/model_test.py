import torch
from kakaoGPT.train import tokenizer,device,model

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        tokens = tokenizer.encode(q,return_tensors='pt').to(device=device,non_blocking=True)
        gen_tokens = model.generate(tokens, do_smaple=True, tenperature=0.8, max_length=64)
        generated = tokenizer.batch_decode(gen_tokens)[0]

        print("Chatbot > {}".format(generated))
        

