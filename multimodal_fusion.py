# multimodal_fusion.py
import json
from typing import Dict, List

class MultimodalFusion:
    def __init__(self):
        self.modality_weights = {
            'text': 1.0,
            'figure': 0.7,
            'formula': 0.8,
            'table': 0.9
        }
    
    def build_multimodal_context(self, paper_dir: str) -> str:
        context_parts = []
        
        summary_path = os.path.join(paper_dir, f"{os.path.basename(paper_dir)}_summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                context_parts.append(f"[TEXT_SUMMARY]\n{f.read().strip()}\n[/TEXT_SUMMARY]")
        
        formula_path = os.path.join(paper_dir, f"{os.path.basename(paper_dir)}_formulas.json")
        if os.path.exists(formula_path):
            with open(formula_path, 'r', encoding='utf-8') as f:
                formulas = json.load(f)
                if formulas:
                    formula_text = "\n".join([f"- {f.get('content', '')}" for f in formulas[:5]])  
                    context_parts.append(f"[FORMULAS]\n{formula_text}\n[/FORMULAS]")
        
        tables_dir = os.path.join(paper_dir, "tables")
        if os.path.exists(tables_dir):
            table_files = [f for f in os.listdir(tables_dir) if f.endswith('.png')][:3]  
            if table_files:
                context_parts.append(f"[TABLES]\n发现 {len(table_files)} 个表格\n[/TABLES]")
        
        figures_dir = os.path.join(paper_dir, "figures")
        if os.path.exists(figures_dir):
            figure_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')][:5]  # 限制数量
            if figure_files:
                context_parts.append(f"[FIGURES]\n发现 {len(figure_files)} 个图表\n[/FIGURES]")
        
        return "\n\n".join(context_parts)
    
    def enhance_rag_retrieval(self, query: str, multimodal_data: Dict) -> str:
        enhanced_query = query
        
        if any(keyword in query.lower() for keyword in ['formula', 'equation', '公式', '方程']):
            if 'formulas' in multimodal_data:
                enhanced_query += f"\n相关公式: {multimodal_data['formulas'][:2]}"
        
        if any(keyword in query.lower() for keyword in ['table', '表格', '数据']):
            if 'tables' in multimodal_data:
                enhanced_query += f"\n相关表格: {multimodal_data['tables'][:2]}"
        
        return enhanced_query