# table_processor.py
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import pytesseract

class TableProcessor:
    def __init__(self):
        pass
    
    def extract_table_structure(self, table_image_path):
        """提取表格结构为Markdown"""
        try:
            img = cv2.imread(table_image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用OCR识别表格内容
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # 简单重构为表格（实际项目中可使用更复杂的表格检测）
            texts = [data['text'][i] for i in range(len(data['text'])) if data['text'][i].strip()]
            
            # 尝试按行分组
            rows = []
            current_row = []
            last_top = -1
            
            for i in range(len(data['text'])):
                if not data['text'][i].strip():
                    continue
                top = data['top'][i]
                if last_top == -1 or abs(top - last_top) < 10:
                    current_row.append(data['text'][i])
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [data['text'][i]]
                last_top = top
            
            if current_row:
                rows.append(current_row)
            
            # 转换为Markdown
            if rows:
                header = "| " + " | ".join(rows[0]) + " |"
                separator = "|" + "|".join(["---"] * len(rows[0])) + "|"
                body = "\n".join(["| " + " | ".join(row) + " |" for row in rows[1:]])
                return f"{header}\n{separator}\n{body}"
            return ""
        except Exception as e:
            print(f"表格解析失败 {table_image_path}: {e}")
            return ""
    
    def process_table_directory(self, tables_dir):
        """处理tables文件夹"""
        results = {}
        if not os.path.exists(tables_dir):
            return results
        
        for fname in os.listdir(tables_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(tables_dir, fname)
                markdown = self.extract_table_structure(path)
                results[fname] = {
                    'markdown': markdown,
                    'path': path
                }
        return results