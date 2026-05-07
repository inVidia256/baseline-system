import os
import torch
import re
from typing import List, Dict, Tuple
import base64
from PIL import Image
import io

class QwenImageImportanceSelector:
    """
    使用 Qwen3-VL 模型评估图像重要性
    完全基于 Qwen3-VL，不使用 CLIP
    """
    def __init__(self, model_server):
        """
        初始化图像重要性评估器
        :param model_server: 已初始化的 ModelServer 实例
        """
        self.server = model_server
        print("🖼️ 初始化 Qwen3-VL 图像重要性评估器...")
        
        # 定义图像类型关键词
        self.type_keywords = {
            "architecture": ["architecture", "framework", "structure", "model", "network", "diagram"],
            "results": ["result", "accuracy", "performance", "comparison", "experiment", "evaluation"],
            "data": ["data", "distribution", "statistics", "dataset", "sample"],
            "process": ["process", "flow", "workflow", "pipeline", "methodology"],
            "table": ["table", "tab", "data sheet", "comparison table"],
            "graph": ["graph", "plot", "chart", "curve", "trend", "visualization"]
        }
        
        # 定义重要性评分提示词模板
        self.importance_prompt_template = """请作为学术论文评审专家，评估以下图像对理解论文核心内容的贡献度。

论文摘要：{abstract}

图像描述：{image_description}

请根据以下标准评估该图像的重要性（0.0-1.0分）：
1. 核心贡献展示（0.4分）：是否展示论文的核心创新点或主要贡献
2. 实验验证支撑（0.3分）：是否提供关键的实验结果或数据支撑
3. 方法流程说明（0.2分）：是否清晰展示研究方法或技术路线
4. 对比分析价值（0.1分）：是否提供有意义的对比或分析视角

请只返回一个0.0到1.0之间的数字，不要任何其他文字。"""

    def analyze_image_filename(self, image_path: str) -> Dict[str, float]:
        """
        基于文件名分析图像类型和重要性
        """
        filename = os.path.basename(image_path).lower()
        
        type_scores = {
            "architecture": 0.0,
            "results": 0.0,
            "data": 0.0,
            "process": 0.0,
            "table": 0.0,
            "graph": 0.0
        }
        
        # 根据文件名关键词判断类型
        for img_type, keywords in self.type_keywords.items():
            for keyword in keywords:
                if keyword in filename:
                    type_scores[img_type] = 0.9
                    break
        
        # 如果没有匹配，默认为结果图
        if all(score == 0.0 for score in type_scores.values()):
            type_scores["results"] = 0.7
            
        return type_scores
    
    def get_image_description(self, image_path: str, abstract: str) -> str:
        """
        使用 Qwen3-VL 生成图像描述
        """
        try:
            # 准备提示词
            prompt = f"""请简要描述这张学术论文中的图像，重点关注：
1. 这是什么类型的图像（架构图、结果图、数据图、流程图等）
2. 图像展示了什么内容
3. 对理解论文有什么帮助

论文摘要参考：{abstract[:500]}"""
            
            # 使用 Qwen3-VL 生成描述
            description = self.server.generate_summary_with_images(
                system_prompt="你是一个学术论文图像分析专家。",
                user_prompt=prompt,
                image_paths=[image_path],
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9
            )
            
            return description.strip()
        except Exception as e:
            print(f"⚠️ 图像描述生成失败 ({image_path}): {e}")
            return "学术论文中的一张图像"
    
    def evaluate_importance_with_qwen(self, image_path: str, abstract: str, image_description: str) -> float:
        """
        使用 Qwen3-VL 评估图像重要性
        """
        try:
            # 构建评估提示词
            prompt = self.importance_prompt_template.format(
                abstract=abstract[:1000],
                image_description=image_description[:500]
            )
            
            # 使用 Qwen3-VL 进行评估
            response = self.server.generate_summary_with_images(
                system_prompt="你是一个严格的学术论文评审专家，擅长评估图像对论文理解的价值。",
                user_prompt=prompt,
                image_paths=[image_path],
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.9
            )
            
            # 提取数字分数
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                # 确保在0-1范围内
                score = max(0.0, min(1.0, score))
                return score
            else:
                # 如果没找到数字，尝试解析文本
                if "重要" in response or "核心" in response:
                    return 0.8
                elif "有用" in response or "支撑" in response:
                    return 0.6
                else:
                    return 0.4
                    
        except Exception as e:
            print(f"⚠️ 重要性评估失败 ({image_path}): {e}")
            return 0.5
    
    def calculate_composite_score(self, image_path: str, abstract: str, 
                               position_weight: float = 1.0) -> float:
        """
        计算综合重要性分数
        """
        # 1. 基于文件名的类型分数
        type_scores = self.analyze_image_filename(image_path)
        type_weight = max(type_scores.values())
        
        # 2. 使用 Qwen 生成图像描述
        image_description = self.get_image_description(image_path, abstract)
        
        # 3. 使用 Qwen 评估重要性
        importance_score = self.evaluate_importance_with_qwen(
            image_path, abstract, image_description
        )
        
        # 4. 基于文件大小的复杂度分数
        try:
            file_size = os.path.getsize(image_path)
            complexity_score = min(1.0, file_size / (1024 * 500))  # 500KB为满分
        except:
            complexity_score = 0.5
        
        # 5. 综合分数
        composite_score = (
            0.4 * importance_score +    # Qwen评估的重要性
            0.3 * type_weight +         # 图像类型权重
            0.2 * position_weight +      # 位置权重
            0.1 * complexity_score      # 复杂度权重
        )
        
        return composite_score, image_description
    
    def select_key_images(self, image_paths: List[str], abstract: str, 
                         top_k: int = 2, min_score_threshold: float = 0.4) -> List[Tuple[str, float, str]]:
        """
        选择最重要的图像
        
        Returns:
            List of (image_path, score, description)
        """
        if not image_paths:
            return []
        
        scored_images = []
        
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue
                
            # 位置权重：越靠前的图像通常越重要
            position_weight = 1.0 - (idx / len(image_paths)) * 0.5
            
            # 计算综合分数
            score, description = self.calculate_composite_score(
                img_path, abstract, position_weight
            )
            
            if score >= min_score_threshold:
                scored_images.append((img_path, score, description))
        
        # 按分数降序排序
        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个
        selected = scored_images[:top_k]
        
        print(f"📊 Qwen3-VL 图像重要性评估完成:")
        for img_path, score, desc in selected:
            print(f"   {os.path.basename(img_path)}: {score:.3f} - {desc[:50]}...")
        
        return selected
    
    def batch_select_images(self, image_groups: Dict[str, List[str]], abstract: str, 
                          max_total: int = 3) -> List[str]:
        """
        从多个图像组中智能选择图像
        """
        all_selected = []
        
        # 为每组图像选择最重要的
        for group_name, paths in image_groups.items():
            if not paths:
                continue
                
            # 每组最多选的数量
            group_max = 2 if group_name in ["figures", "tables"] else 1
            
            selected = self.select_key_images(
                paths, abstract, top_k=group_max
            )
            
            all_selected.extend(selected)
        
        # 如果总数超过限制，重新排序选择
        if len(all_selected) > max_total:
            all_selected.sort(key=lambda x: x[1], reverse=True)
            all_selected = all_selected[:max_total]
        
        # 只返回路径
        return [path for path, _, _ in all_selected]


# 测试函数
if __name__ == "__main__":
    from serve import ModelServer
    
    # 初始化模型服务器
    server = ModelServer(model_name="/workspace/work/models/Qwen3-VL-8B-Instruct", quantize="int8")
    
    # 初始化图像选择器
    selector = QwenImageImportanceSelector(server)
    
    # 测试单张图像
    test_image = "papers/sample/figures/figure_001.png"
    if os.path.exists(test_image):
        abstract = "This paper proposes a novel framework for..."
        score, description = selector.calculate_composite_score(
            test_image, abstract, position_weight=1.0
        )
        print(f"测试图像重要性分数: {score:.3f}")
        print(f"图像描述: {description[:100]}...")