import json
import jsonlines
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from latex2sympy2 import latex2sympy
from sympy import simplify, Eq
from fractions import Fraction

CNT=1
#——————————————————————————————————————————————————evaluate
def extract_boxed_answer(solution_str):
    # 先用一个简单的正则找到 \boxed{ 的起始位置
    match_start = re.search(r'\\boxed{', solution_str)
    if not match_start:
        return None
    
    start_index = match_start.end()
    level = 1
    
    for i in range(start_index, len(solution_str)):
        char = solution_str[i]
        if char == '{':
            level += 1
        elif char == '}':
            level -= 1
        
        if level == 0:
            # 找到了与之匹配的结束括号
            return solution_str[start_index:i]
            
    return None # 没有找到匹配的结束括号

def normalize_math(s):
    """深度数学规范化处理"""
    if not s:
        return ''
    # 移除所有空格和格式字符
    s = re.sub(r'\s+|\\,|\\quad|\\qquad|\\!|\\;', '', s)
    s = re.sub(r',', '', s)  # 移除千分位逗号  
    
    # # 处理等式格式（x=5 → 5）
    # s = re.sub(r'^.*?=', '', s)  # 保留等号右边内容  
       
    # 统一分数格式
    s = re.sub(r'\\(d|t)frac\b', r'\\frac', s)  # 关键修改点
    s = re.sub(r'\\frac(\d+)(\d+)', r'\\frac{\1}{\2}', s)  # \frac43 → \frac{4}{3}

    # 统一选项格式
    s = re.sub(r'^([A-Z])$', r'\\text{(\1)}', s)
    
    return s.strip()

def adjust_extracted_format(extracted_norm, answer_norm):
    """根据答案格式调整提取结果格式"""
    # 处理\text{...}情况
    if re.fullmatch(r'\\text\{([^{}]+)\}', answer_norm) and '\\text' not in extracted_norm:
        return f'\\text{{{extracted_norm}}}'
    
    # 处理$情况
    if answer_norm.startswith('\\$') and '\\$' not in extracted_norm:
        return f'\\${extracted_norm}'
    
    # 处理^\circ情况
    if '^\\circ' in answer_norm and '^\\circ' not in extracted_norm:
        return f'{extracted_norm}^\\circ'
    
    return extracted_norm

def is_math_equivalent(expr1, expr2):
    if len(expr1) > 50 or len(expr2) > 50:
        return False
    try:
        expr1_sympy = latex2sympy(expr1)
        expr2_sympy = latex2sympy(expr2)
        return float(expr1_sympy) == float(expr2_sympy)
    except:
        return False

def evaluate_math(solution,answer):
    # 提取候选答案
    global CNT
    #print(CNT)
    extracted = extract_boxed_answer(solution) 
    CNT+=1    
    # 深度规范化
    answer_norm = normalize_math(answer)
    extracted_norm = normalize_math(extracted) if extracted else None
    if extracted_norm is None:
        return False
    if extracted_norm and answer_norm:
        extracted_norm = adjust_extracted_format(extracted_norm, answer_norm)                
    if extracted_norm.lower()==answer_norm.lower():
        return True
    # 数学等价性判断
    if is_math_equivalent(extracted_norm, answer_norm):
        return True
    
    return False
    


if __name__ == "__main__":
        
    # 1. 定义原始数据集的文件名和路径（需根据实际路径调整）
    dataset_config = {
        "olympiad": "/data/sy/projects/LongKV/data/olympiad.jsonl",
        "minerva": "/data/sy/projects/LongKV/data/minerva.jsonl",
        "gpqa": "/data/sy/projects/LongKV/data/gpqa.jsonl",
        "aime25": "/data/sy/projects/LongKV/data/aime25.jsonl",
        "amc": "/data/sy/projects/LongKV/data/amc.jsonl",
        "math": "/data/sy/projects/LongKV/data/math.jsonl",
        "aime24": "/data/sy/projects/LongKV/data/aime24.jsonl",
        "gsm8k": "/data/sy/projects/LongKV/data/gsm8k.jsonl"
    }

    # 2. 预处理：读取每个原始数据集的条目，存储为可匹配的集合（元组形式）
    dataset_entries = {}
    for dataset_name, file_path in dataset_config.items():
        entries = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 提取唯一标识：index + question + true_answer（确保与baseline的字段一致）
                key = (data["index"], data["question"])
                entries.add(key)
        dataset_entries[dataset_name] = entries

    # 3. 处理baseline.jsonl，统计每个数据集的数量
    baseline_path = "/data/sy/projects/LongKV/outputs/qwen_8k_2k/ours_nopenalty.jsonl"
    # counts = {name: 0 for name in dataset_config}
    counts = {
        name: {"total": 0, "correct": 0, "accuracy": 0.0}
        for name in dataset_config
    }    
    with open(baseline_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 构建匹配键（与原始数据集的键一致）
            match_key = (data["index"], data["question"])
            # import pdb;pdb.set_trace()
            # 遍历所有数据集，找到匹配项
            for dataset_name, entries in dataset_entries.items():
                if match_key in entries:
                    counts[dataset_name]["total"] += 1
                    if evaluate_math(data["generated_answer"],str(data["true_answer"])):
                        counts[dataset_name]["correct"] += 1
                    break  # 找到后立即跳出，避免重复匹配
    # 4. 输出结果            
    print("数据集统计结果：")
    for dataset, stats in counts.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        print(f"{dataset:>8}:", end="\t")
        print(f"  总数量: {stats['total']:>8}", end="\t")
        print(f"  正确数量: {stats['correct']:>8}", end="\t")
        print(f"  准确率: {stats['accuracy']:.4f}", end="\t")  # 保留4位小数
        print()
        
        # for dataset, count in counts.items():
        #     print(f"{dataset}: {count}")x