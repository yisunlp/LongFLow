import json
from tqdm import tqdm
import re
from latex2sympy2 import latex2sympy
from sympy import simplify
import multiprocessing # 导入多进程模块

baseline_path = "/data/sy/projects/LongKV/outputs/qwen_8k_2k/baseline.jsonl"
num_processes = 32
#——————————————————————————————————————————————————
# evaluate 函数（保持不变，子进程会调用它们）
#——————————————————————————————————————————————————

def extract_boxed_answer(solution_str):
    # 先用一个简单的正则找到 \boxed{ 的起始位置
    match_start = re.search(r'\\boxed{', solution_str)
    if not match_start:
        return None
    
    start_index = match_start.end()
    level = 1
    
    # 稍微优化一下，如果字符串很长，预先计算长度
    str_len = len(solution_str)
    for i in range(start_index, str_len):
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
    
    # 统一分数格式
    s = re.sub(r'\\(d|t)frac\b', r'\\frac', s)
    s = re.sub(r'\\frac(\d+)(\d+)', r'\\frac{\1}{\2}', s)

    # 统一选项格式
    s = re.sub(r'^([A-Z])$', r'\\text{(\1)}', s)
    
    return s.strip()

def adjust_extracted_format(extracted_norm, answer_norm):
    """根据答案格式调整提取结果格式"""
    if re.fullmatch(r'\\text\{([^{}]+)\}', answer_norm) and '\\text' not in extracted_norm:
        return f'\\text{{{extracted_norm}}}'
    if answer_norm.startswith('\\$') and '\\$' not in extracted_norm:
        return f'\\${extracted_norm}'
    if '^\\circ' in answer_norm and '^\\circ' not in extracted_norm:
        return f'{extracted_norm}^\\circ'
    return extracted_norm

def is_math_equivalent(expr1, expr2):
    """数学等价性判断"""
    if len(expr1) > 50 or len(expr2) > 50:
        return False
    try:
        # 使用 Sympy 的 simplify 进行更可靠的比较
        expr1_sympy = latex2sympy(expr1)
        expr2_sympy = latex2sympy(expr2)
        
        # simplify(expr1 - expr2) == 0 是一个更鲁棒的判断相等的方式
        if simplify(expr1_sympy - expr2_sympy) == 0:
            return True
    except Exception:
        # 如果转换或简化失败，回退到字符串比较
        pass
        
    # 作为最后的手段，进行原始字符串比较
    return expr1 == expr2

def evaluate_math(solution, answer):
    """评估主函数"""
    extracted = extract_boxed_answer(solution) 
    answer_norm = normalize_math(answer)
    extracted_norm = normalize_math(extracted) if extracted else None
    
    if extracted_norm is None:
        return False
    
    if extracted_norm and answer_norm:
        extracted_norm = adjust_extracted_format(extracted_norm, answer_norm)
        
    if extracted_norm.lower() == answer_norm.lower():
        return True
        
    if is_math_equivalent(extracted_norm, answer_norm):
        return True
    
    return False

#——————————————————————————————————————————————————
# 多进程工作函数 (Worker Function)
#——————————————————————————————————————————————————
def process_line(line, dataset_entries):
    """
    处理单行 baseline 数据。
    这是每个子进程要执行的核心任务。
    """
    data = json.loads(line)
    match_key = (data["index"], data["question"])
    
    for dataset_name, entries in dataset_entries.items():
        if match_key in entries:
            is_correct = evaluate_math(data["generated_answer"], str(data["true_answer"]))
            return (dataset_name, is_correct) # 返回结果元组
            
    return (None, False) # 如果没有在任何数据集中找到匹配项

#——————————————————————————————————————————————————
# 主程序
#——————————————————————————————————————————————————
if __name__ == "__main__":
    # 冻结支持，在某些操作系统（如Windows）上是必需的
    multiprocessing.freeze_support()

    # 1. 定义数据集配置
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

    # 2. 预处理：读取原始数据集（这个部分很快，无需并行）
    print("正在预加载数据集元数据...")
    dataset_entries = {}
    for dataset_name, file_path in dataset_config.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries = set()
                for line in f:
                    data = json.loads(line)
                    key = (data.get("index"), data.get("question"))
                    entries.add(key)
                dataset_entries[dataset_name] = entries
        except FileNotFoundError:
            print(f"警告: 找不到文件 {file_path}，将跳过该数据集。")
            dataset_entries[dataset_name] = set()

    # 3. 设置多进程并处理 baseline.jsonl
    counts = {
        name: {"total": 0, "correct": 0, "accuracy": 0.0}
        for name in dataset_config
    }

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            lines = f.readlines() # 将所有行读入内存
        
        # 定义进程数，通常设置为CPU核心数或稍小一些

        print(f"使用 {num_processes} 个进程进行并行处理...")

        # 创建进程池
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用 imap_unordered 来分发任务并获取结果
            # 使用 functools.partial 来固定 dataset_entries 参数
            from functools import partial
            worker_func = partial(process_line, dataset_entries=dataset_entries)
            
            # 使用 tqdm 显示进度条
            results = list(tqdm(pool.imap_unordered(worker_func, lines), total=len(lines), desc="评估进度"))

        # 4. 聚合结果
        print("正在聚合结果...")
        for dataset_name, is_correct in results:
            if dataset_name and dataset_name in counts:
                counts[dataset_name]["total"] += 1
                if is_correct:
                    counts[dataset_name]["correct"] += 1
    
    except FileNotFoundError:
        print(f"错误: 找不到 baseline 文件: {baseline_path}")
        exit()

    # 5. 计算并输出最终结果
    print("\n数据集统计结果：")
    for dataset, stats in counts.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        print(f"{dataset:>8}:", end="\t")
        print(f"  总数量: {stats['total']:>8}", end="\t")
        print(f"  正确数量: {stats['correct']:>8}", end="\t")
        print(f"  准确率: {stats['accuracy']:.4f}", end="\t")
        print()