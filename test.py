import os
import json
import time
import sys
import torch

from text_chunker import chunk_text
from rag_system import EnhancedRAGSystem
from serve import ModelServer
from evaluator import evaluate_summary
from prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def load_multimodal_info(paper_dir, paper_id):
    multimodal_info = {
        'formulas': [],   # 公式的LaTeX字符串列表
        'tables': [],     # 表格图片路径列表
        'figures': []     # 图表图片路径列表
    }
    
    formula_path = os.path.join(paper_dir, f"{paper_id}_formulas.json")
    if os.path.exists(formula_path):
        try:
            with open(formula_path, "r", encoding="utf-8") as f:
                formula_data = json.load(f)
            formulas = formula_data.get("data", [])
            for formula in formulas[:5]: 
                latex = formula.get("latex", "")
                if latex:
                    multimodal_info['formulas'].append(latex)
        except Exception as e:
            print(f"加载公式失败: {e}")
    
    table_dir = os.path.join(paper_dir, "tables")
    if os.path.exists(table_dir):
        table_files = [os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith('.png')]
        multimodal_info['tables'] = table_files[:2]  
    
    figure_dir = os.path.join(paper_dir, "figures")
    if os.path.exists(figure_dir):
        figure_files = [os.path.join(figure_dir, f) for f in os.listdir(figure_dir) if f.endswith('.png')]
        multimodal_info['figures'] = figure_files[:2]  
    return multimodal_info

def find_text_json(paper_dir):
    for file in os.listdir(paper_dir):
        if file.endswith("_text.json"):
            return os.path.join(paper_dir, file)
    for file in os.listdir(paper_dir):
        if file.endswith(".json") and not file.endswith("ABSTRACT.json"):
            return os.path.join(paper_dir, file)
    return None

def load_paper_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("data", [])
    full_text = " ".join([c["content"] for c in chunks])
    
    base_dir = os.path.dirname(json_path)
    abstract_path = os.path.join(base_dir, "ABSTRACT.json")
    reference_summary = ""
    if os.path.exists(abstract_path):
        with open(abstract_path, "r", encoding="utf-8") as f:
            reference_summary = json.load(f).get("content", "")
    return full_text, reference_summary

def generate_summary_for_mode(rag, server, full_text, mode_name, image_paths=None):
    print(f"\n🔍 {mode_name}: 执行混合检索...")
    query = full_text[:1000]
    
    key_chunks = rag.hybrid_retrieve(
        query=query,
        vector_k=4,
        bm25_k=4,
        rerank_k=2
    )
    
    context = "\n\n".join(key_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(text=context)
    
    print(f"{mode_name}: 正在生成摘要...")
    start_time = time.time()
    
    if image_paths:
        summary = server.generate_summary_with_images(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=image_paths,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
    else:
        summary = server.generate_summary(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
    
    elapsed = time.time() - start_time
    print(f"⏱ {mode_name} 生成耗时: {elapsed:.2f}秒")
    
    return summary, elapsed

def test_sample_workflow():
    sample_dir = "papers/sample"
    model_name = "/workspace/work/models/Qwen3-VL-8B-Instruct"
    
    if not os.path.exists(sample_dir):
        print(f"错误: {sample_dir} 目录不存在")
        return False
    
    text_json = find_text_json(sample_dir)
    if not text_json:
        print(f"错误: 在 {sample_dir} 中未找到文本JSON文件")
        return False
    
    print(f"测试文件: {os.path.basename(text_json)}")
    
    try:
        server = ModelServer(model_name=model_name, quantize="int8")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    full_text, reference_summary = load_paper_json(text_json)
    
    if not full_text.strip():
        print("❌ 错误: 文本内容为空")
        return False
    
    paper_id = os.path.basename(sample_dir)
    multimodal_info = load_multimodal_info(sample_dir, paper_id)
    
    print(f"  多模态信息: {list(multimodal_info.keys())}")
    
    text_chunks = chunk_text(full_text, chunk_size=1024, overlap=200)
    
    print("\n" + "="*60)
    print("模式1: 纯文本RAG")
    print("="*60)
    
    rag_text_only = EnhancedRAGSystem()
    rag_text_only.clear_collection()
    rag_text_only.add_documents(text_chunks)
    
    summary_text_only, time_text_only = generate_summary_for_mode(
        rag_text_only, server, full_text, "纯文本RAG"
    )
    
    print(f"📝 纯文本摘要: {summary_text_only[:200]}...")
    
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("模式2: 多模态RAG（使用图像）")
    print("="*60)
    
    rag_multimodal = EnhancedRAGSystem()
    rag_multimodal.clear_collection()
    rag_multimodal.add_documents(text_chunks)  
    
    image_paths = []
    if multimodal_info['tables']:
        image_paths.extend(multimodal_info['tables'][:1])
    if multimodal_info['figures']:
        image_paths.extend(multimodal_info['figures'][:1])
    image_paths = image_paths[:2]  
    
    summary_multimodal, time_multimodal = generate_summary_for_mode(
        rag_multimodal, server, full_text, "多模态RAG", image_paths
    )
    
    print(f"多模态摘要: {summary_multimodal[:200]}...")
    
    if reference_summary:
        print("\n" + "="*60)
        print("ROUGE 分数对比")
        print("="*60)
        
        metrics_text = evaluate_summary(reference_summary, summary_text_only)
        print(f"\n纯文本RAG vs 参考摘要:")
        print(f"  ROUGE-1: {metrics_text['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics_text['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics_text['rougeL']:.4f}")
        print(f"  BLEU:    {metrics_text['bleu']:.4f}")
        
        metrics_multi = evaluate_summary(reference_summary, summary_multimodal)
        print(f"\n多模态RAG vs 参考摘要:")
        print(f"  ROUGE-1: {metrics_multi['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics_multi['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics_multi['rougeL']:.4f}")
        print(f"  BLEU:    {metrics_multi['bleu']:.4f}")
        
        print(f"\n多模态相对纯文本的改进:")
        rouge1_improve = (metrics_multi['rouge1'] - metrics_text['rouge1']) / metrics_text['rouge1'] * 100 if metrics_text['rouge1'] > 0 else 0
        rougeL_improve = (metrics_multi['rougeL'] - metrics_text['rougeL']) / metrics_text['rougeL'] * 100 if metrics_text['rougeL'] > 0 else 0
        bleu_improve = (metrics_multi['bleu'] - metrics_text['bleu']) / metrics_text['bleu'] * 100 if metrics_text['bleu'] > 0 else 0
        
        print(f"  ROUGE-1 改进: {rouge1_improve:+.2f}%")
        print(f"  ROUGE-L 改进: {rougeL_improve:+.2f}%")
        print(f"  BLEU 改进:    {bleu_improve:+.2f}%")
        
        print(f"\n多模态RAG vs 纯文本RAG:")
        metrics_vs = evaluate_summary(summary_text_only, summary_multimodal)
        print(f"  ROUGE-1: {metrics_vs['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics_vs['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics_vs['rougeL']:.4f}")
        print(f"  BLEU:    {metrics_vs['bleu']:.4f}")
        
        comparison_result = {
            "reference_summary": reference_summary,
            "summary_text_only": summary_text_only,
            "summary_multimodal": summary_multimodal,
            "time_text_only": time_text_only,
            "time_multimodal": time_multimodal,
            "metrics_text": {
                "rouge1": metrics_text['rouge1'],
                "rouge2": metrics_text['rouge2'],
                "rougeL": metrics_text['rougeL'],
                "bleu": metrics_text['bleu']
            },
            "metrics_multimodal": {
                "rouge1": metrics_multi['rouge1'],
                "rouge2": metrics_multi['rouge2'],
                "rougeL": metrics_multi['rougeL'],
                "bleu": metrics_multi['bleu']
            },
            "improvement_percentage": {
                "rouge1": rouge1_improve,
                "rougeL": rougeL_improve,
                "bleu": bleu_improve
            }
        }
        
        output_file = os.path.join(sample_dir, "comparison_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n对比结果已保存至: {output_file}")
    
    return True

if __name__ == "__main__":
    success = test_sample_workflow()
    sys.exit(0 if success else 1)