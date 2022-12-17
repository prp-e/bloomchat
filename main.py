from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b3", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b3")
