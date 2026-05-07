"""
测试 Qwen3-VL 模型在训练模式下的行为
"""
import torch
import os
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def test_qwen3vl_safe():
    
    model_path = "/workspace/work/models/Qwen3-VL-8B-Instruct"
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=" * 80)
    print("测试 1: 模型加载和基本属性")
    print("=" * 80)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print(f"模型类型: {type(model)}")
    print(f"模型设备: {model.device}")
    print(f"模型训练模式: {model.training}")
    
    print("\n" + "=" * 80)
    print("测试 2: 创建极小测试输入")
    print("=" * 80)
    
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,  
        return_dict=True
    ).to(model.device)
    
    print(f"输入键: {list(inputs.keys())}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    
    print("\n" + "=" * 80)
    print("测试 3: 推理模式")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"推理模式输出键: {list(outputs.keys())}")
    if hasattr(outputs, 'loss'):
        print(f"推理模式 loss: {outputs.loss}")
    else:
        print("推理模式: 没有 loss")
    
    print("\n" + "=" * 80)
    print("测试 4: 训练模式（启用梯度检查点）")
    print("=" * 80)
    
    model.gradient_checkpointing_enable()
    model.train()
    
    try:
        with torch.amp.autocast('cuda'):
            outputs_train = model(**inputs)
        
        print(f"训练模式输出键: {list(outputs_train.keys())}")
        
        if hasattr(outputs_train, 'loss') and outputs_train.loss is not None:
            print(f"训练模式 loss: {outputs_train.loss}")
            print(f"loss 是否有梯度: {outputs_train.loss.requires_grad}")
        else:
            print("训练模式: 没有 loss，需要手动计算")
            
            if "logits" in outputs_train:
                logits = outputs_train["logits"]
                print(f"logits shape: {logits.shape}")
                print(f"logits 是否需要梯度: {logits.requires_grad}")
                
                labels = inputs["input_ids"].clone()
                pad_token_id = processor.tokenizer.pad_token_id
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100
                
                loss_fct = torch.nn.CrossEntropyLoss()
                labels = labels.to(logits.device)
                
                if not logits.requires_grad:
                    print("警告: logits 不需要梯度，可能无法反向传播")
                    with torch.enable_grad():
                        outputs_retry = model(**inputs)
                        logits = outputs_retry.get("logits")
                
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                print(f"手动计算的 loss: {loss}")
                print(f"手动 loss 是否需要梯度: {loss.requires_grad}")
                
                try:
                    loss.backward()
                    print("✓ 反向传播成功")
                except Exception as e:
                    print(f"✗ 反向传播失败: {e}")
            else:
                print("没有 logits，无法计算 loss")
    
    except torch.OutOfMemoryError as e:
        print(f"✗ OOM 错误: {e}")
        print("建议：进一步减少序列长度或使用更激进的量化")
    
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("综合结论")
    print("=" * 80)
    print("1. 模型默认不在训练模式")
    print("2. 推理模式不返回 loss")
    print("3. 训练模式也不返回 loss，需要手动计算")
    print("4. 需要手动处理梯度和反向传播")
    
    return True

if __name__ == "__main__":
    try:
        test_qwen3vl_safe()
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()