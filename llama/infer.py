from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)