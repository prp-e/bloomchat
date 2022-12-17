from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

set_seed(424242) # I think it's not necessary at all 😁

prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "
input_ids = tokenizer(prompt, return_tensors="pt").to(0)
sample = model.generate(**input_ids, max_length=50, top_k=0, temperature=0.7)
print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
