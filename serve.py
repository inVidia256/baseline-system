import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import base64
import os

class ModelServer:
    def __init__(self, model_name=None, quantize="int8", device="cuda"):
        if model_name is None:
            model_name = "/workspace/work/models/Qwen3-VL-8B-Instruct"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # 构建量化配置
        bnb_config = None
        if quantize == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        load_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        print(f"🚀 Loading model: {model_name} (quantize: {quantize})")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def _encode_image_to_base64(self, image_path):
        """将图片转换为 Base64 编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_summary(self, system_prompt, user_prompt,
                         max_new_tokens=512, temperature=0.3, top_p=0.9):
        """
        纯文本摘要生成（无图像），使用官方推荐 processor.apply_chat_template 流程。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # 使用 processor 进行 tokenize（官方方式）
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # 截取生成的 token（去掉输入部分），与官方示例一致
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

    def generate_summary_with_images(self, system_prompt, user_prompt, image_paths=None,
                                   max_new_tokens=512, temperature=0.3, top_p=0.9):
        """
        支持图像输入的摘要生成。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 添加文本部分
        messages[1]["content"].append({"type": "text", "text": user_prompt})
        
        # 添加图像部分（限制数量，防止爆显存）
        if image_paths:
            for img_path in image_paths[:2]:  # 最多2张图片
                if os.path.exists(img_path):
                    base64_image = self._encode_image_to_base64(img_path)
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
                else:
                    print(f"⚠️ 图像路径不存在: {img_path}")
        
        # 使用 processor 进行 tokenize（官方方式）
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # 截取生成的 token（去掉输入部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()