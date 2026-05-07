
import os
import json
import re
import time
import csv
from datetime import datetime
from text_chunker import chunk_text
from rag_system import EnhancedRAGSystem
from serve import ModelServer
from evaluator import evaluate_summary
from prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def clean_ocr_text(text):
    patterns = [
        r"^The text content from the image is:\s*",
        r"^The text content from the image is as follows:\s*",
        r"^The image contains a single line of text that reads:\s*",
        r"^Here is the text content from the image:\s*",
        r"^The text extracted from the image is:\s*",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()

def load_paper_from_sample(sample_dir):
    text_file = None
    for fname in os.listdir(sample_dir):
        if fname.endswith("_text.json") or fname == "sample_text.json":
            text_file = os.path.join(sample_dir, fname)
            break
    if text_file is None:
        for fname in os.listdir(sample_dir):
            if fname.endswith(".json") and not fname.endswith("ABSTRACT.json"):
                text_file = os.path.join(sample_dir, fname)
                break
    if text_file is None:
        raise FileNotFoundError("在 sample 目录中未找到合适的文本 JSON 文件")

    with open(text_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "data" in data and isinstance(data["data"], list):
        full_text = " ".join([clean_ocr_text(chunk.get("content", "")) for chunk in data["data"]])
    elif "content" in data:
        full_text = clean_ocr_text(data["content"])
    elif isinstance(data, str):
        full_text = clean_ocr_text(data)
    else:
        full_text = ""

    abstract_file = os.path.join(sample_dir, "ABSTRACT.json")
    reference_summary = ""
    if os.path.exists(abstract_file):
        with open(abstract_file, "r", encoding="utf-8") as f:
            abstract_data = json.load(f)
        reference_summary = abstract_data.get("content", "")

    return full_text, reference_summary

def generate_intelligent_query(text, num_sentences=3):
    sentences = re.split(r'(?<=[。！？?!\.])(?![a-zA-Z]\w)|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= num_sentences:
        return text
    start_count = max(1, num_sentences // 2)
    start_sents = sentences[:start_count]
    end_count = num_sentences - start_count
    end_sents = sentences[-end_count:] if end_count > 0 else []
    return " ".join(start_sents + end_sents)

def run_experiment(config, full_text, reference_summary, server, experiment_id):
    print(f"\n{'='*60}\n实验 #{experiment_id}: {config['name']}\n{'='*60}")
    start_time = time.time()
    
    rag = EnhancedRAGSystem()
    rag.clear_collection()
    
    text_chunks = chunk_text(
        full_text,
        chunk_size=config['CHUNK_SIZE'],
        overlap=config['OVERLAP'],
        truncate_to=config['TRUNCATE_TO'],
        language="english"
    )
    
    rag.add_documents(text_chunks)
    query = generate_intelligent_query(full_text, num_sentences=3)
    
    key_chunks = rag.hybrid_retrieve(
        query=query,
        vector_k=config['VECTOR_K'],
        bm25_k=config['BM25_K'],
        rerank_k=config['RETRIEVE_K']
    )
    
    context = "\n\n---\n\n".join(key_chunks) if key_chunks else ""
    user_prompt = USER_PROMPT_TEMPLATE.format(text=context)
    summary = server.generate_summary(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=config['MAX_NEW_TOKENS'],
        temperature=config['TEMPERATURE'],
        top_p=config['TOP_P']
    )
    
    elapsed = time.time() - start_time
    metrics = evaluate_summary(reference_summary, summary)
    
    result = {
        'experiment_id': experiment_id,
        'config_name': config['name'],
        'summary': summary,
        'summary_length': len(summary),
        'reference_length': len(reference_summary),
        'time_seconds': elapsed,
        'rouge1': metrics['rouge1'],
        'rouge2': metrics['rouge2'],
        'rougeL': metrics['rougeL'],
        'bleu': metrics['bleu'],
    }
    for key, value in config.items():
        result[f'config_{key}'] = value
    
    return result

def save_results_to_csv(results, filename):
    if not results: return
    fieldnames = list(results[0].keys())
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def generate_massive_param_combinations(base_config):
    combinations = []
    exp_id = 1
    
    for cs in [1024, 2048, 4096, 8192, 12288]:
        for tr_ratio in [0.2, 0.5, 0.8]:
            config = base_config.copy()
            config['CHUNK_SIZE'] = cs
            config['TRUNCATE_TO'] = int(cs * tr_ratio)
            config['OVERLAP'] = int(cs * 0.15)
            config['name'] = f"grid_chunk_{cs}_tr{tr_ratio}"
            combinations.append(config)
            exp_id += 1

    for k in [1, 2, 4, 6, 8, 10, 12, 15, 20, 30]:
        for pool_mult in [1.5, 3, 5]: 
            config = base_config.copy()
            config['RETRIEVE_K'] = k
            config['VECTOR_K'] = int(k * pool_mult)
            config['BM25_K'] = int(k * pool_mult)
            config['name'] = f"grid_k_{k}_poolx{pool_mult}"
            combinations.append(config)
            exp_id += 1

    hybrid_ratios = [
        (40, 5), (30, 10), (20, 20), (10, 30), (5, 40)
    ]
    for vk, bk in hybrid_ratios:
        config = base_config.copy()
        config['VECTOR_K'] = vk
        config['BM25_K'] = bk
        config['RETRIEVE_K'] = 8
        config['name'] = f"bias_v{vk}_b{bk}"
        combinations.append(config)
        exp_id += 1

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        for p in [0.5, 0.7, 0.85, 0.95, 1.0]:
            config = base_config.copy()
            config['TEMPERATURE'] = t
            config['TOP_P'] = p
            config['name'] = f"gen_t{t}_p{p}"
            combinations.append(config)
            exp_id += 1

    c_dense = base_config.copy()
    c_dense.update({'CHUNK_SIZE': 512, 'RETRIEVE_K': 20, 'TRUNCATE_TO': 400, 'name': "extreme_dense_small"})
    combinations.append(c_dense)
    c_sparse = base_config.copy()
    c_sparse.update({'CHUNK_SIZE': 32768, 'RETRIEVE_K': 1, 'TRUNCATE_TO': 5000, 'name': "extreme_sparse_large"})
    combinations.append(c_sparse)

    return combinations

def main():
    
    SAMPLE_DIR = "papers/sample"
    MODEL_PATH = "/workspace/work/models/Qwen3-VL-8B-Instruct"
    
    base_config = {
        'CHUNK_SIZE': 8192,
        'OVERLAP': 800,
        'MAX_NEW_TOKENS': 512,
        'TEMPERATURE': 0.5,
        'TOP_P': 0.9,
        'TRUNCATE_TO': 1000,
        'RETRIEVE_K': 4,
        'VECTOR_K': 8,
        'BM25_K': 8,
    }
    
    if not os.path.isdir(SAMPLE_DIR): return
    full_text, reference_summary = load_paper_from_sample(SAMPLE_DIR)
    
    server = ModelServer(model_name=MODEL_PATH, quantize="int8")
    param_combinations = generate_massive_param_combinations(base_config)
    
    print(f"扫描范围已确定：共计 {len(param_combinations)} 组实验")
    
    results = []
    best_rouge1 = 0
    best_config = None
    best_result = None

    for i, config in enumerate(param_combinations):
        try:
            result = run_experiment(config, full_text, reference_summary, server, i+1)
            results.append(result)
            
            if result['rouge1'] > best_rouge1:
                best_rouge1 = result['rouge1']
                best_config = config.copy()
                best_result = result
                
            if (i + 1) % 10 == 0:
                save_results_to_csv(results, f"tuning_progress_{i+1}.csv")
                
        except Exception as e:
            print(f"实验 {i+1} 出错: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_to_csv(results, f"final_scan_results_{ts}.csv")
    
    if best_config:
        print(f"\n扫描完成！最优组合 ROUGE-1: {best_rouge1:.4f}, 配置: {best_config['name']}")

if __name__ == "__main__":
    main()
