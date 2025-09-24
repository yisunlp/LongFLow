import os
import json
import torch
import numpy as np
import multiprocessing as mp
from transformers import AutoTokenizer, Qwen3ForCausalLM
from tqdm import tqdm
import time
from queue import Empty

# 导入您自定义的 KV Cache 替换模块
from models.replace import replace_vatp

# --- 配置参数 ---
MODEL_NAME = "/data/sy/projects/huggingface_download/models/Qwen3-4B"
INPUT_FILE = "outputs/qwen_8k_2k/baseline.jsonl"
OUTPUT_FILE = "outputs/qwen_8k_2k/vatp.jsonl"
Cache = replace_vatp()

# 这是“每个GPU”内部处理时的小批量大小
WORKER_BATCH_SIZE = 16
# 条件推理的Token长度阈值
TOKEN_THRESHOLD = 2048

def load_and_filter_data(filepath: str, tokenizer: AutoTokenizer, threshold: int):
    """
    从指定文件加载数据，并根据'generated_answer'的token长度进行筛选。
    """
    to_process = []
    to_copy = []
    print(f"[*] 正在从 '{filepath}' 加载并筛选数据...")
    
    # 首先获取文件总行数以用于tqdm
    with open(filepath, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="筛选数据"):
            item = json.loads(line)
            # 确保 'generated_answer' 字段存在
            answer_text = item.get('generated_answer', '')
            
            # Tokenize 并检查长度
            token_ids = tokenizer(answer_text, return_tensors='pt').input_ids
            
            if token_ids.shape[1] < threshold:
                to_copy.append(item)
            else:
                to_process.append(item)

    print(f"[*] 筛选完成。需要重新推理: {len(to_process)} 条, 直接复制: {len(to_copy)} 条。")
    return to_process, to_copy

def inference_worker(gpu_id: int, data_chunk: list, model_name: str, result_queue: mp.Queue):
    """
    在指定GPU上使用 'vatp' KV cache 运行推理的子进程函数。
    """
    device = f"cuda:{gpu_id}"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager" # 或 flash_attention_2
        ).to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        for i in range(0, len(data_chunk), WORKER_BATCH_SIZE):
            # 2. 【核心改动】为每个批次重置/重新初始化 cache
            past_key_values = Cache(hh_size=1024, recent_size=1024)
            
            batch = data_chunk[i:i + WORKER_BATCH_SIZE]
            
            # 注意：这里我们用'question'字段来生成新的答案
            prompts = [item['question'] for item in batch]
            texts = []
            for prompt in prompts:
                SYSTEM_PROMPT = "Please reason step by step, and must put your final answer within \\boxed{}"
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
            
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=1000,truncation=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=8192, # 根据需要调整
                    do_sample=False,
                    past_key_values=past_key_values,
                    use_cache=True,
                    repetition_penalty=1.2
                )

            input_ids_len = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_ids_len:].cpu()
            generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for idx, answer in enumerate(generated_answers):
                # 返回完整的原始item，并用新生成的答案覆盖'generated_answer'字段
                result_item = batch[idx].copy()
                result_item['generated_answer'] = answer.strip()
                result_queue.put(result_item)
                
    except Exception as e:
        print(f"[Worker-{gpu_id}] 发生错误: {e}")
    finally:
        print(f"[Worker-{gpu_id}] 已完成任务。")
        result_queue.put(None) 


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("错误：未检测到任何GPU。")
        return
    
    # 数据处理需要分词器，所以先加载
    print("[*] 正在加载分词器用于数据预处理...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 加载并筛选数据
    to_process, to_copy = load_and_filter_data(INPUT_FILE, tokenizer, TOKEN_THRESHOLD)
    
    # 如果没有需要处理的数据，则直接合并保存并退出
    if not to_process:
        print("[*] 没有需要重新推理的数据。直接保存结果。")
        final_results = sorted(to_copy, key=lambda x: x.get('index', 0))
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in final_results:
                f.write(json.dumps(item) + '\n')
        print(f"[*] 全部完成！结果已保存至 '{OUTPUT_FILE}'。")
        return

    total_items_to_process = len(to_process)
    data_chunks = [chunk.tolist() for chunk in np.array_split(to_process, num_gpus)]
    
    for i, chunk in enumerate(data_chunks):
        print(f"GPU {i} 将处理 {len(chunk)} 条数据。")

    result_queue = mp.Queue()
    processes = []

    print("\n[*] 正在为每张GPU启动 'vatp' 推理进程...")
    start_time = time.time()
    
    for i in range(num_gpus):
        process = mp.Process(target=inference_worker, args=(i, data_chunks[i], MODEL_NAME, result_queue))
        processes.append(process)
        process.start()

    processed_results = []
    done_workers = 0 
    
    with tqdm(total=total_items_to_process, desc="推理进度") as pbar:
        while done_workers < num_gpus:
            try:
                result = result_queue.get(timeout=10000) # 60秒超时
                if result is None:
                    done_workers += 1
                else:
                    processed_results.append(result)
                    pbar.update(1)
            except Empty:
                pbar.write("!!!! 警告：等待超时！可能存在已崩溃的子进程。 !!!!")
                break 

    pbar.close()
    print(f"\n[*] 推理循环结束。{done_workers}/{num_gpus} 个工作进程已发送完成信号。")

    print("[*] 正在清理所有子进程...")
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            print(f"警告：进程 {process.pid} 无法正常join，将强制终止。")
            process.terminate()
    
    # 合并结果
    final_results = to_copy + processed_results
    print(f"[*] 共收集到 {len(final_results)} 条总结果。正在整理并保存...")
    final_results.sort(key=lambda x: x.get('index', 0))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    end_time = time.time()
    print(f"\n[*] 全部完成！结果已保存至 '{OUTPUT_FILE}'。")
    print(f"[*] 总耗时: {end_time - start_time:.2f} 秒。")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()