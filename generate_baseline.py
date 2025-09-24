import os
import json
import torch
import numpy as np
import multiprocessing as mp
from transformers import AutoTokenizer, Qwen3ForCausalLM
from tqdm import tqdm
import time

# --- 配置参数 ---
MODEL_NAME = "/data/sy/projects/huggingface_download/models/Qwen3-4B"
DATA_DIR = "data"
# 确保输出目录存在
os.makedirs("outputs", exist_ok=True)
OUTPUT_FILE = "outputs/qwen_8k_2k/baseline.jsonl"
# 这是“每个GPU”内部处理时的小批量大小，以控制显存占用
WORKER_BATCH_SIZE = 8

def load_all_data(directory: str) -> list:
    """从指定目录加载所有 .jsonl 文件的数据。"""
    all_data = []
    print(f"[*] 从 '{directory}' 文件夹加载数据...")
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"警告：跳过在 {filepath} 中的无效行: {line}")
    print(f"[*] 加载完成，共计 {len(all_data)} 条数据。")
    return all_data

def inference_worker(gpu_id: int, data_chunk: list, model_name: str, result_queue: mp.Queue):
    """
    在指定GPU上运行推理的子进程函数。(此函数无需改动)
    
    Args:
        gpu_id (int): 要使用的GPU的索引 (0, 1, 2, ...)。
        data_chunk (list): 这个进程需要处理的数据子集。
        model_name (str): 模型的路径。
        result_queue (mp.Queue): 用于返回结果的进程安全队列。
    """
    device = f"cuda:{gpu_id}"
    # 为了减少终端输出的杂乱，可以只让一个worker打印启动信息
    if gpu_id == 0:
        print(f"[Worker-{gpu_id}] 已启动，将在 {device} 上运行。(其他worker将静默启动)")

    try:
        # 1. 在指定的GPU上加载模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # 2. 对分给自己的数据进行批量推理
        for i in range(0, len(data_chunk), WORKER_BATCH_SIZE):
            batch = data_chunk[i:i + WORKER_BATCH_SIZE]
            
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
                    max_new_tokens=8192,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.2
                )

            input_ids_len = model_inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_ids_len:].cpu()
            generated_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # 3. 将结果放入队列
            for idx, answer in enumerate(generated_answers):
                result_item = batch[idx].copy()
                result_item['generated_answer'] = answer.strip()
                result_queue.put(result_item)
                
    except Exception as e:
        print(f"[Worker-{gpu_id}] 发生错误: {e}")
    finally:
        print(f"[Worker-{gpu_id}] 已完成任务。(其他worker将静默完成)")


def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("错误：未检测到任何GPU。")
        return
        
    all_data = load_all_data(DATA_DIR)
    total_items = len(all_data)
    
    data_chunks = [chunk.tolist() for chunk in np.array_split(all_data, num_gpus)]
    
    for i, chunk in enumerate(data_chunks):
        print(f"GPU {i} 将处理 {len(chunk)} 条数据。")

    result_queue = mp.Queue()
    processes = []

    print("\n[*] 正在为每张GPU启动一个推理进程...")
    start_time = time.time()
    
    for i in range(num_gpus):
        process = mp.Process(target=inference_worker, args=(i, data_chunks[i], MODEL_NAME, result_queue))
        processes.append(process)
        process.start()

    # ==================== 进度条和结果收集逻辑 (核心改动) ====================
    final_results = []
    # 创建一个tqdm进度条，总数为全部数据的数量
    with tqdm(total=total_items, desc="整体进度") as pbar:
        # 持续循环，直到收集到的结果数量等于总任务数
        while len(final_results) < total_items:
            # 从队列中获取一个结果
            result = result_queue.get()
            final_results.append(result)
            # 每获取一个结果，进度条就更新一次
            pbar.update(1)

    print("\n[*] 所有推理任务已完成。正在清理子进程...")

    # 等待所有进程完成（此时它们应该已经完成了自己的工作）
    for process in processes:
        process.join()
        
    print("[*] 所有子进程已清理完毕。正在整理并保存结果...")
    # ======================== 改动结束 ========================
    
    # (可选但推荐) 按原始索引排序
    final_results.sort(key=lambda x: x.get('index', 0))

    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item) + '\n')
            
    end_time = time.time()
    print(f"\n[*] 全部完成！共处理 {len(final_results)} 条结果并保存至 '{OUTPUT_FILE}'。")
    print(f"[*] 总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()