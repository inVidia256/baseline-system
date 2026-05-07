# formula_processor.py
import json
import re
from PIL import Image
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

class FormulaProcessor:
    def __init__(self):
        print("🔄 加载公式识别模型...")
        self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-latex-ocr")
        self.model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-latex-ocr")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def latex_to_readable(self, latex):
        replacements = {
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'\1/\2',
            r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
            r'\\sum_\{([^}]+)\}\^\{([^}]+)\}': r'sum from \1 to \2',
            r'\\int_\{([^}]+)\}\^\{([^}]+)\}': r'integral from \1 to \2',
            r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ',
            r'\\delta': 'δ', r'\\epsilon': 'ε', r'\\theta': 'θ',
        }
        text = latex
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)
        return text
    
    def process_formula_json(self, formula_json_path):
        if not os.path.exists(formula_json_path):
            return []
        
        with open(formula_json_path, 'r', encoding='utf-8') as f:
            formulas = json.load(f)
        
        processed = []
        for formula in formulas:
            latex = formula.get('content', '')
            readable = self.latex_to_readable(latex)
            processed.append({
                'raw_latex': latex,
                'readable': readable,
                'page': formula.get('page', 0)
            })
        return processed