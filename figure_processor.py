# figure_processor.py
import json
import os
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class FigureProcessor:
    def __init__(self, use_captioning=True):
        self.use_captioning = use_captioning
        if use_captioning:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def extract_figure_text(self, image_path):
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            return text.strip()
        except Exception as e:
            print(f"OCR失败 {image_path}: {e}")
            return ""
    
    def generate_caption(self, image_path):
        if not self.use_captioning:
            return ""
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(img, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"图像描述失败 {image_path}: {e}")
            return ""
    
    def process_figure_directory(self, figures_dir):
        results = {}
        if not os.path.exists(figures_dir):
            return results
        
        for fname in os.listdir(figures_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(figures_dir, fname)
                ocr_text = self.extract_figure_text(path)
                caption = self.generate_caption(path)
                results[fname] = {
                    'ocr_text': ocr_text,
                    'caption': caption,
                    'path': path
                }
        return results