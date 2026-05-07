"""
使用 LoRA 技术微调 Qwen3-VL-8B 模型
结构化摘要生成
从 YAML 配置文件读取超参数
"""

import os
import json
import yaml
import torch
import time  
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from text_chunker import chunk_text
    from evaluator import evaluate_summary
    from prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
except ImportError:
    SYSTEM_PROMPT = "你是一个专业的学术编辑，擅长撰写结构化摘要。"
    USER_PROMPT_TEMPLATE = "请根据以下内容生成结构化摘要：\n{text}"
    
    def chunk_text(text, chunk_size=1024, overlap=200):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
    
    def evaluate_summary(reference, generated):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

def setup_logging(config: Dict[str, Any]):
    """根据配置设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config['logging']['log_file'].format(timestamp=timestamp)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "finetune_config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['logging']['log_file'] = config['logging']['log_file'].format(timestamp=timestamp)
    
    return config

@dataclass
class FinetuneConfig:
    """从 YAML 配置创建配置对象"""
    def __init__(self, config_dict: Dict[str, Any]):
        # 模型配置
        self.model_name = config_dict['model']['name']
        self.output_dir = config_dict['model']['output_dir']
        
        # 量化配置
        self.load_in_8bit = config_dict['model']['load_in_8bit']
        self.load_in_4bit = config_dict['model']['load_in_4bit']
        
        # LoRA 配置
        self.lora_r = config_dict['lora']['r']
        self.lora_alpha = config_dict['lora']['alpha']
        self.lora_dropout = config_dict['lora']['dropout']
        self.target_modules = config_dict['lora']['target_modules']
        
        # 训练配置
        self.num_train_epochs = config_dict['training']['num_train_epochs']
        self.per_device_train_batch_size = config_dict['training']['per_device_train_batch_size']
        self.gradient_accumulation_steps = config_dict['training']['gradient_accumulation_steps']
        self.learning_rate = config_dict['training']['learning_rate']
        self.warmup_steps = config_dict['training']['warmup_steps']
        self.weight_decay = config_dict['training']['weight_decay']
        self.max_seq_length = config_dict['training']['max_seq_length']
        self.save_strategy = config_dict['training']['save_strategy']
        self.save_total_limit = config_dict['training']['save_total_limit']
        self.logging_steps = config_dict['training']['logging_steps']
        self.optim = config_dict['training']['optim']
        self.fp16 = config_dict['training']['fp16']
        self.dataloader_pin_memory = config_dict['training']['dataloader_pin_memory']
        
        # 数据集配置
        self.papers_dir = config_dict['dataset']['papers_dir']
        self.max_samples = config_dict['dataset']['max_samples']
        self.validation_split = config_dict['dataset']['validation_split']
        
        # 提示词模板
        self.system_message = config_dict['prompt_template']['system_message']
        self.structure = config_dict['prompt_template']['structure']
        self.format_template = config_dict['prompt_template']['format']
        
        # 日志配置
        self.log_level = config_dict['logging']['log_level']
        self.log_file = config_dict['logging']['log_file']
        self.tensorboard_dir = config_dict['logging']['tensorboard_dir']
        self.report_to = config_dict['logging']['report_to']
        
        # 测试配置
        self.test_after_training = config_dict['testing']['test_after_training']
        self.test_paper_dir = config_dict['testing']['test_paper_dir']
        self.max_test_length = config_dict['testing']['max_test_length']
        self.save_test_result = config_dict['testing']['save_test_result']
        self.test_output_file = config_dict['testing']['test_output_file']

def load_paper_data(config: FinetuneConfig, logger) -> List[Dict[str, str]]:
    """从 papers 文件夹加载论文数据"""
    logger.info(f"开始从 {config.papers_dir} 加载论文数据...")
    
    papers_data = []
    
    if not os.path.exists(config.papers_dir):
        logger.error(f"目录不存在: {config.papers_dir}")
        return papers_data
    
    for paper_folder in os.listdir(config.papers_dir):
        if len(papers_data) >= config.max_samples:
            break
            
        folder_path = os.path.join(config.papers_dir, paper_folder)
        if not os.path.isdir(folder_path):
            continue
            
        text_file = None
        for file in os.listdir(folder_path):
            if file.endswith("_text.json") or file == "sample_text.json":
                text_file = os.path.join(folder_path, file)
                break
        
        if not text_file:
            continue
            
        abstract_file = os.path.join(folder_path, "ABSTRACT.json")
        if not os.path.exists(abstract_file):
            continue
            
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
            
            if "data" in text_data and isinstance(text_data["data"], list):
                full_text = " ".join([chunk.get("content", "") for chunk in text_data["data"]])
            elif "content" in text_data:
                full_text = text_data["content"]
            else:
                full_text = str(text_data)
            
            with open(abstract_file, 'r', encoding='utf-8') as f:
                abstract_data = json.load(f)
                summary = abstract_data.get("content", "")
            
            if full_text and summary:
                papers_data.append({
                    "text": full_text[:config.max_seq_length * 2], 
                    "summary": summary
                })
                logger.info(f"成功加载论文: {paper_folder}")
                
        except Exception as e:
            logger.warning(f"加载论文 {paper_folder} 失败: {e}")
            continue
    
    logger.info(f"共加载 {len(papers_data)} 篇论文数据")
    return papers_data

def create_structured_prompt(text: str, config: FinetuneConfig) -> str:
    """创建结构化提示词"""
    return config.format_template.format(
        background=config.structure['background'],
        method=config.structure['method'],
        experiments=config.structure['experiments'],
        conclusion=config.structure['conclusion'],
        text=text
    )

def create_instruction_dataset(papers_data: List[Dict[str, str]], config: FinetuneConfig) -> Dataset:
    """创建指令微调数据集"""
    logger.info("创建指令微调数据集...")
    
    data = []
    for paper in papers_data:
        instruction = create_structured_prompt(paper["text"], config)
        data.append({
            "instruction": instruction,
            "input": "",
            "output": paper["summary"]
        })
    
    dataset = Dataset.from_list(data)
    logger.info(f"数据集创建完成，共 {len(dataset)} 条样本")
    return dataset

def setup_model_and_lora(config: FinetuneConfig, logger):
    """设置模型和 LoRA 配置 - 显存优化版"""
    logger.info(f"加载模型: {config.model_name}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model.gradient_checkpointing_enable()
    
    logger.info("配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()
    
    return model, processor

def preprocess_function(examples, processor, max_length: int):
    """数据预处理函数 - 修复版本"""
    instructions = examples["instruction"]
    outputs = examples["output"]
    
    conversations = []
    for instruction, output in zip(instructions, outputs):
        conversation = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        conversations.append(conversation)
    
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_dict=True
    )
    
    if isinstance(inputs, dict):
        if "input_ids" in inputs:
            labels = inputs["input_ids"].clone()
            pad_token_id = None
            if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'pad_token_id'):
                pad_token_id = processor.tokenizer.pad_token_id
            elif hasattr(processor, 'pad_token_id'):
                pad_token_id = processor.pad_token_id
            
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            
            inputs["labels"] = labels
    
    return inputs

def train_model(config: FinetuneConfig, logger):
    """主训练函数 - 修复版本"""
    logger.info("=" * 80)
    logger.info("开始 LoRA 微调 Qwen3-VL-8B 模型")
    logger.info("=" * 80)
    
    papers_data = load_paper_data(config, logger)
    if not papers_data:
        logger.error("没有找到可用的论文数据，退出训练")
        return None, None
    
    dataset = create_instruction_dataset(papers_data, config)
    model, processor = setup_model_and_lora(config, logger)
    
    logger.info("预处理数据集...")
    
    def safe_preprocess_function(examples):
        return preprocess_function(examples, processor, min(config.max_seq_length, 512))
    
    tokenized_dataset = dataset.map(
        safe_preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="预处理数据"
    )
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=1,  
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=1,  
        learning_rate=1e-4,
        warmup_steps=0,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",  
        fp16=False,  
        bf16=False,  
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    from transformers import Trainer
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            model.train()
            
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            loss = None
            if labels is not None:
                logits = outputs.get("logits")
                if logits is not None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    labels = labels.to(logits.device)
                    
                    if not logits.requires_grad:
                        with torch.enable_grad():
                            outputs = model(**inputs)
                            logits = outputs.get("logits")
                    
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"警告: 损失值为 {loss}")
                        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            if loss is None:
                device = next(model.parameters()).device
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            outputs["loss"] = loss
            
            if return_outputs:
                return loss, outputs
            return loss
        
        def training_step(self, model, inputs, num_items_in_batch=None):
            """重写 training_step，手动处理反向传播"""
            model.train()
            
            labels = inputs.pop("labels", None)
            
            outputs = model(**inputs)
            
            loss = None
            if labels is not None:
                logits = outputs.get("logits")
                if logits is not None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    labels = labels.to(logits.device)
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
            
            if loss is None:
                device = next(model.parameters()).device
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            loss.backward()
            
            return loss.detach()
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        ),
    )
    
    logger.info("开始训练...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"训练失败: {e}")
        torch.cuda.empty_cache()
        raise
    
    logger.info("保存微调后的模型...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    
    logger.info(f"训练完成！模型已保存到: {config.output_dir}")
    
    return model, processor

def load_multimodal_info(paper_dir, paper_id):
    """
    从 JSON 文件和文件系统中加载多模态信息，并返回文件路径。
    参考 test.py 中的实现
    """
    multimodal_info = {
        'formulas': [],   
        'tables': [],     
        'figures': []     
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
            logger.warning(f"⚠️ 加载公式失败: {e}")
    
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
    """在论文目录中查找文本JSON文件"""
    for file in os.listdir(paper_dir):
        if file.endswith("_text.json"):
            return os.path.join(paper_dir, file)
    for file in os.listdir(paper_dir):
        if file.endswith(".json") and not file.endswith("ABSTRACT.json"):
            return os.path.join(paper_dir, file)
    return None

def load_paper_json(json_path):
    """加载论文文本JSON文件"""
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

def test_finetuned_model(model, processor, config: FinetuneConfig, logger):
    """测试微调后的模型 - 参考 test.py 的实现"""
    if not config.test_after_training:
        logger.info("跳过测试")
        return
    
    logger.info("测试微调后的模型...")
    
    test_paper_dir = config.test_paper_dir
    if not os.path.exists(test_paper_dir):
        logger.warning(f"测试目录不存在: {test_paper_dir}")
        return
    text_json = find_text_json(test_paper_dir)
    if not text_json:
        logger.warning(f"在 {test_paper_dir} 中未找到文本JSON文件")
        return
    
    logger.info(f"📄 测试文件: {os.path.basename(text_json)}")
    
    full_text, reference_summary = load_paper_json(text_json)
    
    if not full_text.strip():
        logger.warning("错误: 文本内容为空")
        return
    
    paper_id = os.path.basename(test_paper_dir)
    multimodal_info = load_multimodal_info(test_paper_dir, paper_id)
    logger.info(f"多模态信息: {list(multimodal_info.keys())}")
    
    text_chunks = chunk_text(full_text, chunk_size=1024, overlap=200)
    context = text_chunks[0] if text_chunks else full_text[:1000]
    user_prompt = USER_PROMPT_TEMPLATE.format(text=context)
    
    image_paths = []
    if multimodal_info['tables']:
        image_paths.extend(multimodal_info['tables'][:1])
    if multimodal_info['figures']:
        image_paths.extend(multimodal_info['figures'][:1])
    image_paths = image_paths[:2]  
    
    logger.info("🧠 正在生成摘要...")
    start_time = time.time()  
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    if image_paths:
        messages[-1]["images"] = image_paths
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start_time
    
    logger.info(f"⏱️ 生成耗时: {elapsed:.2f}秒")
    logger.info("📝 生成的摘要:")
    logger.info(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
    
    if reference_summary:
        logger.info("\n" + "="*60)
        logger.info("📊 ROUGE 分数评估")
        logger.info("="*60)
        
        metrics = evaluate_summary(reference_summary, generated_text)
        
        logger.info(f"ROUGE-1: {metrics.get('rouge1', 0.0):.4f}")
        logger.info(f"ROUGE-2: {metrics.get('rouge2', 0.0):.4f}")
        logger.info(f"ROUGE-L: {metrics.get('rougeL', 0.0):.4f}")
        logger.info(f"BLEU: {metrics.get('bleu', 0.0):.4f}")
        
        evaluation_result = {
            "reference_summary": reference_summary,
            "generated_summary": generated_text,
            "generation_time": elapsed,
            "metrics": metrics,
            "multimodal_info": {
                "num_tables": len(multimodal_info['tables']),
                "num_figures": len(multimodal_info['figures']),
                "num_formulas": len(multimodal_info['formulas'])
            }
        }
        
        if config.save_test_result:
            test_output_file = os.path.join(config.output_dir, config.test_output_file)
            with open(test_output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 评估结果已保存到: {test_output_file}")
    else:
        if config.save_test_result:
            test_output_file = os.path.join(config.output_dir, config.test_output_file)
            with open(test_output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            logger.info(f"💾 生成的摘要已保存到: {test_output_file}")

def main():
    """主函数"""
    config_dict = load_config("finetune_config.yaml")
    config = FinetuneConfig(config_dict)
    
    global logger
    logger = setup_logging(config_dict)
    
    try:
        model, processor = train_model(config, logger)
        
        if model and processor:
            test_finetuned_model(model, processor, config, logger)
            
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
    
    logger.info("程序结束")

if __name__ == "__main__":
    main()