# multimodal_utils.py
# 整合：OCR文字识别 + LaTeX公式识别 + PDF表格提取(Markdown)

import os
import cv2
import pytesseract
from PIL import Image
from pix2tex.cli import LatexOCR
import camelot
from typing import List, Dict, Optional


class MultimodalProcessor:
    def __init__(self):
        print("初始化多模态处理器...")
        try:
            self.latex_ocr = LatexOCR()
            print("LaTeX-OCR 加载成功")
        except Exception as e:
            print(f"LaTeX-OCR 加载失败: {e}")
            self.latex_ocr = None

    def ocr_text_from_image(self, image_path: str) -> str:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='eng+chi_sim')
            return text.strip()
        except Exception as e:
            return f"[OCR Text Error: {e}]"

    def ocr_formula_from_image(self, image_path: str) -> str:
        if self.latex_ocr is None:
            return "[LaTeX OCR Not Available]"
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "[Image Read Error]"
            latex = self.latex_ocr(img)
            return f"$${latex.strip()}$$" if latex else ""
        except Exception as e:
            return f"[LaTeX OCR Error: {e}]"

    def process_single_image(self, image_path: str) -> Dict[str, str]:
        return {
            "type": "image",
            "path": image_path,
            "text": self.ocr_text_from_image(image_path),
            "formula": self.ocr_formula_from_image(image_path)
        }

    def batch_process_images(self, image_dir: str) -> List[Dict[str, str]]:
        results = []
        if not os.path.isdir(image_dir):
            print(f"⚠️ 目录不存在: {image_dir}")
            return results
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(valid_exts):
                full_path = os.path.join(image_dir, fname)
                results.append(self.process_single_image(full_path))
        print(f"处理了 {len(results)} 张图片")
        return results

    def extract_tables_to_markdown(self, pdf_path: str, pages: str = "all") -> List[str]:
        markdown_tables = []
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")
            
            for i, table in enumerate(tables):
                df = table.df
                df.columns = df.iloc[0]
                df = df[1:]
                md = df.to_markdown(index=False)
                markdown_tables.append(md)
            
            print(f" 从 {pdf_path} 提取了 {len(markdown_tables)} 个表格")
        except Exception as e:
            print(f" 表格提取失败 ({pdf_path}): {e}")
        
        return markdown_tables

    def collect_multimodal_chunks(self, pdf_path: str, image_dir: Optional[str] = None) -> List[str]:
        chunks = []

        tables = self.extract_tables_to_markdown(pdf_path)
        for i, table_md in enumerate(tables):
            chunk = f"[TABLE START]\nTable {i+1}:\n{table_md}\n[TABLE END]"
            chunks.append(chunk)

        if image_dir and os.path.isdir(image_dir):
            images_info = self.batch_process_images(image_dir)
            for info in images_info:
                chunk = (
                    f"[FIGURE START]\n"
                    f"Image Path: {info['path']}\n"
                    f"Text Content: {info['text']}\n"
                    f"Formula: {info['formula']}\n"
                    f"[FIGURE END]"
                )
                chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    processor = MultimodalProcessor()
    
    test_pdf = "papers/sample/paper.pdf"
    test_img_dir = "papers/sample/images"
    
    if os.path.exists(test_pdf):
        chunks = processor.collect_multimodal_chunks(test_pdf, test_img_dir)
        for c in chunks:
            print("-" * 50)
            print(c[:500]) 