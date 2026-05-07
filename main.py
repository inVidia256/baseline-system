import os
import json
import time
import sys
import torch
import glob
import csv
from typing import List, Dict, Any
from datetime import datetime

from text_chunker import chunk_text
from rag_system import EnhancedRAGSystem
from serve import ModelServer
from evaluator import evaluate_summary
from prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def find_text_json(paper_dir):
    for file in os.listdir(paper_dir):
        if file.endswith("_text.json"):
            return os.path.join(paper_dir, file)
    
    sample_text = os.path.join(paper_dir, "sample_text.json")
    if os.path.exists(sample_text):
        return sample_text
    
    for file in os.listdir(paper_dir):
        if file.endswith(".json") and not file.endswith("ABSTRACT.json"):
            return os.path.join(paper_dir, file)
    return None

def load_paper_json(json_path):
    """加载论文文本JSON文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "data" in data and isinstance(data["data"], list):
        chunks = data["data"]
        full_text = " ".join([c.get("content", "") for c in chunks])
    elif "content" in data:
        full_text = data["content"]
    elif isinstance(data, str):
        full_text = data
    else:
        full_text = " ".join([str(v) for v in data.values() if isinstance(v, str)])
    
    base_dir = os.path.dirname(json_path)
    abstract_path = os.path.join(base_dir, "ABSTRACT.json")
    reference_summary = ""
    if os.path.exists(abstract_path):
        with open(abstract_path, "r", encoding="utf-8") as f:
            abstract_data = json.load(f)
            reference_summary = abstract_data.get("content", "")
    
    return full_text, reference_summary

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
            print(f"⚠️ 加载公式失败: {e}")
    
    table_dir = os.path.join(paper_dir, "tables")
    if os.path.exists(table_dir):
        table_files = glob.glob(os.path.join(table_dir, "*.png"))
        multimodal_info['tables'] = table_files[:2]  
    
    figure_dir = os.path.join(paper_dir, "figures")
    if os.path.exists(figure_dir):
        figure_files = glob.glob(os.path.join(figure_dir, "*.png"))
        multimodal_info['figures'] = figure_files[:2]  
    
    return multimodal_info

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
    
    print(f"🧠 {mode_name}: 正在生成摘要...")
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

def process_single_paper(paper_dir, server):
    paper_id = os.path.basename(paper_dir)
    print(f"\n{'='*80}")
    print(f"📄 处理论文: {paper_id}")
    print(f"{'='*80}")
    
    text_json = find_text_json(paper_dir)
    if not text_json:
        print(f"❌ 错误: 在 {paper_dir} 中未找到文本JSON文件")
        return None
    
    print(f"📄 文本文件: {os.path.basename(text_json)}")
    
    full_text, reference_summary = load_paper_json(text_json)
    
    if not full_text.strip():
        print("❌ 错误: 文本内容为空")
        return None
    
    print(f"📝 文本长度: {len(full_text)} 字符")
    print(f"📋 参考摘要: {'存在' if reference_summary else '不存在'}")
    
    multimodal_info = load_multimodal_info(paper_dir, paper_id)
    print(f"🖼️ 多模态信息: 公式 {len(multimodal_info['formulas'])}, 表格 {len(multimodal_info['tables'])}, 图表 {len(multimodal_info['figures'])}")
    
    text_chunks = chunk_text(full_text, chunk_size=1024, overlap=200)
    print(f"📦 文本分块: {len(text_chunks)} 个块")
    
    result = {
        'paper_id': paper_id,
        'text_length': len(full_text),
        'num_chunks': len(text_chunks),
        'has_reference': bool(reference_summary),
        'num_formulas': len(multimodal_info['formulas']),
        'num_tables': len(multimodal_info['tables']),
        'num_figures': len(multimodal_info['figures'])
    }
    
    print("\n" + "="*60)
    print("模式1: 纯文本RAG")
    print("="*60)
    
    rag_text_only = EnhancedRAGSystem()
    rag_text_only.clear_collection()
    rag_text_only.add_documents(text_chunks)
    
    summary_text_only, time_text_only = generate_summary_for_mode(
        rag_text_only, server, full_text, "纯文本RAG"
    )
    
    result['summary_text_only'] = summary_text_only
    result['time_text_only'] = time_text_only
    
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
    
    result['image_paths_used'] = image_paths
    
    summary_multimodal, time_multimodal = generate_summary_for_mode(
        rag_multimodal, server, full_text, "多模态RAG", image_paths
    )
    
    result['summary_multimodal'] = summary_multimodal
    result['time_multimodal'] = time_multimodal
    
    print(f"📝 多模态摘要: {summary_multimodal[:200]}...")
    
    if reference_summary:
        print("\n" + "="*60)
        print("📊 ROUGE 分数对比")
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
        
        rouge1_improve = ((metrics_multi['rouge1'] - metrics_text['rouge1']) / 
                         metrics_text['rouge1'] * 100) if metrics_text['rouge1'] > 0 else 0
        rouge2_improve = ((metrics_multi['rouge2'] - metrics_text['rouge2']) / 
                         metrics_text['rouge2'] * 100) if metrics_text['rouge2'] > 0 else 0
        rougeL_improve = ((metrics_multi['rougeL'] - metrics_text['rougeL']) / 
                         metrics_text['rougeL'] * 100) if metrics_text['rougeL'] > 0 else 0
        bleu_improve = ((metrics_multi['bleu'] - metrics_text['bleu']) / 
                       metrics_text['bleu'] * 100) if metrics_text['bleu'] > 0 else 0
        
        print(f"\n📈 多模态相对纯文本的改进:")
        print(f"  ROUGE-1 改进: {rouge1_improve:+.2f}%")
        print(f"  ROUGE-2 改进: {rouge2_improve:+.2f}%")
        print(f"  ROUGE-L 改进: {rougeL_improve:+.2f}%")
        print(f"  BLEU 改进:    {bleu_improve:+.2f}%")
        
        result['reference_summary'] = reference_summary
        result['metrics_text'] = metrics_text
        result['metrics_multimodal'] = metrics_multi
        result['improvement_percentage'] = {
            'rouge1': rouge1_improve,
            'rouge2': rouge2_improve,
            'rougeL': rougeL_improve,
            'bleu': bleu_improve
        }
    
    output_file = os.path.join(paper_dir, "comparison_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_file}")
    
    return result

def find_all_paper_dirs(base_dir="papers"):
    paper_dirs = []
    
    if not os.path.exists(base_dir):
        print(f"❌ 错误: 目录 '{base_dir}' 不存在")
        return paper_dirs
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            text_json = find_text_json(item_path)
            if text_json:
                paper_dirs.append(item_path)
    
    return paper_dirs

def save_batch_results_to_csv(all_results, filename="batch_results.csv"):
    if not all_results:
        print("⚠️ 没有结果可保存")
        return
    
    fieldnames = [
        'paper_id', 'text_length', 'num_chunks', 'has_reference',
        'num_formulas', 'num_tables', 'num_figures',
        'time_text_only', 'time_multimodal',
        'text_rouge1', 'text_rouge2', 'text_rougeL', 'text_bleu',
        'multi_rouge1', 'multi_rouge2', 'multi_rougeL', 'multi_bleu',
        'improvement_rouge1', 'improvement_rouge2', 'improvement_rougeL', 'improvement_bleu'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {
                'paper_id': result.get('paper_id', ''),
                'text_length': result.get('text_length', 0),
                'num_chunks': result.get('num_chunks', 0),
                'has_reference': result.get('has_reference', False),
                'num_formulas': result.get('num_formulas', 0),
                'num_tables': result.get('num_tables', 0),
                'num_figures': result.get('num_figures', 0),
                'time_text_only': result.get('time_text_only', 0),
                'time_multimodal': result.get('time_multimodal', 0)
            }
            
            if 'metrics_text' in result:
                row.update({
                    'text_rouge1': result['metrics_text'].get('rouge1', 0),
                    'text_rouge2': result['metrics_text'].get('rouge2', 0),
                    'text_rougeL': result['metrics_text'].get('rougeL', 0),
                    'text_bleu': result['metrics_text'].get('bleu', 0)
                })
            
            if 'metrics_multimodal' in result:
                row.update({
                    'multi_rouge1': result['metrics_multimodal'].get('rouge1', 0),
                    'multi_rouge2': result['metrics_multimodal'].get('rouge2', 0),
                    'multi_rougeL': result['metrics_multimodal'].get('rougeL', 0),
                    'multi_bleu': result['metrics_multimodal'].get('bleu', 0)
                })
            
            if 'improvement_percentage' in result:
                row.update({
                    'improvement_rouge1': result['improvement_percentage'].get('rouge1', 0),
                    'improvement_rouge2': result['improvement_percentage'].get('rouge2', 0),
                    'improvement_rougeL': result['improvement_percentage'].get('rougeL', 0),
                    'improvement_bleu': result['improvement_percentage'].get('bleu', 0)
                })
            
            writer.writerow(row)
    
    print(f"CSV结果已保存至: {filename}")

def save_batch_summary(all_results, filename="batch_summary.json"):
    summary = {
        'total_papers': len(all_results),
        'successful_papers': len([r for r in all_results if 'metrics_text' in r]),
        'failed_papers': len([r for r in all_results if 'metrics_text' not in r]),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'average_scores': {},
        'detailed_results': all_results
    }
    
    if all_results:
        text_metrics = [r['metrics_text'] for r in all_results if 'metrics_text' in r]
        multi_metrics = [r['metrics_multimodal'] for r in all_results if 'metrics_multimodal' in r]
        
        if text_metrics:
            summary['average_scores']['text_only'] = {
                'rouge1': sum(m['rouge1'] for m in text_metrics) / len(text_metrics),
                'rouge2': sum(m['rouge2'] for m in text_metrics) / len(text_metrics),
                'rougeL': sum(m['rougeL'] for m in text_metrics) / len(text_metrics),
                'bleu': sum(m['bleu'] for m in text_metrics) / len(text_metrics)
            }
        
        if multi_metrics:
            summary['average_scores']['multimodal'] = {
                'rouge1': sum(m['rouge1'] for m in multi_metrics) / len(multi_metrics),
                'rouge2': sum(m['rouge2'] for m in multi_metrics) / len(multi_metrics),
                'rougeL': sum(m['rougeL'] for m in multi_metrics) / len(multi_metrics),
                'bleu': sum(m['bleu'] for m in multi_metrics) / len(multi_metrics)
            }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"📋 汇总结果已保存至: {filename}")

def main():    
    MODEL_PATH = "/workspace/work/models/Qwen3-VL-8B-Instruct"
    
    try:
        server = ModelServer(model_name=MODEL_PATH, quantize="int8")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    paper_dirs = find_all_paper_dirs("papers")
    
    print(f"\n找到 {len(paper_dirs)} 篇论文待处理:")
    for i, paper_dir in enumerate(paper_dirs, 1):
        print(f"  {i}. {os.path.basename(paper_dir)}")
    
    all_results = []
    successful = 0
    failed = 0
    
    for i, paper_dir in enumerate(paper_dirs, 1):
        print(f"\n{'#'*80}")
        print(f"进度: {i}/{len(paper_dirs)}")
        print(f"{'#'*80}")
        
        try:
            result = process_single_paper(paper_dir, server)
            if result:
                all_results.append(result)
                successful += 1
                print(f"论文 {os.path.basename(paper_dir)} 处理完成")
            else:
                failed += 1
                print(f"论文 {os.path.basename(paper_dir)} 处理失败")
        except Exception as e:
            print(f"处理论文 {os.path.basename(paper_dir)} 时出错: {e}")
            failed += 1
            continue
        finally:
            torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("批量处理完成")
    print("="*80)
    print(f"成功处理: {successful} 篇")
    print(f"处理失败: {failed} 篇")
    
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"batch_results_{timestamp}.csv"
        save_batch_results_to_csv(all_results, csv_filename)
        
        summary_filename = f"batch_summary_{timestamp}.json"
        save_batch_summary(all_results, summary_filename)
        
        if successful > 0:
            text_avg = all_results[0]['metrics_text']
            multi_avg = all_results[0]['metrics_multimodal']
            
            print(f"\n📈 平均分数对比:")
            print(f"  纯文本RAG - ROUGE-1: {text_avg['rouge1']:.4f}, ROUGE-L: {text_avg['rougeL']:.4f}")
            print(f"  多模态RAG - ROUGE-1: {multi_avg['rouge1']:.4f}, ROUGE-L: {multi_avg['rougeL']:.4f}")
            
            if 'improvement_percentage' in all_results[0]:
                imp = all_results[0]['improvement_percentage']
                print(f"\n多模态相对纯文本的平均改进:")
                print(f"  ROUGE-1: {imp['rouge1']:+.2f}%")
                print(f"  ROUGE-L: {imp['rougeL']:+.2f}%")

if __name__ == "__main__":
    main()