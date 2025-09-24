from transformers import AutoTokenizer,Qwen3ForCausalLM, AutoConfig
from models.replace import replace_sink, replace_h2o, replace_ours, replace_rkv, replace_vatp
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model_name")          
parser.add_argument("--method")            

args = parser.parse_args()
method= args.method
model_name = args.model_name

if method == "ours":
    Cache = replace_ours()
    past_key_values = Cache(cache_budget=1024)

elif method == "sink":
    Cache = replace_sink()
    past_key_values = Cache(num_sink_tokens=32, window_length=992)
elif method == "h2o":
    Cache = replace_h2o()
    past_key_values = Cache(hh_size=512, recent_size=512)
elif method == "rkv":
    Cache = replace_rkv()
    past_key_values = Cache(cache_budget=1024)
elif method == "vatp":
    Cache = replace_vatp()
    past_key_values = Cache(hh_size=512, recent_size=512)
else:
    past_key_values = None

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
model = Qwen3ForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager")
    

texts = []
prompts=["Please provide the proof process of Euler's theorem.",
         "The set of points $(x,y,z)$ that satisfy\n\\[2x = 3y = -z\\]is a line.\n\nThe set of points $(x,y,z)$ that satisfy\n\\[6x = -y = -4z\\]is another line.\n\nFind the angle between these lines, in degrees."]
for prompt in prompts:
    SYSTEM_PROMPT = "Please reason step by step, and must put your final answer within \\boxed{}"
    messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    texts.append(text)

model_inputs = tokenizer(texts, return_tensors="pt", padding="max_length",max_length=2048).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2000,
    do_sample=False,
    past_key_values=past_key_values,
    use_cache=True
)
for i in range(len(prompts)):
    output_ids = generated_ids[i][len(model_inputs.input_ids[0]):].tolist() 
    content = tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
    print("output:", content)
    print("output length:", len(output_ids))

