"""
LLM引擎核心组件
实现Phi-3-mini模型的加载、推理和多模态融合
基于G-Retriever的投影器设计
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import numpy as np
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .config import LLMConfig, Phi3Config
from .input_router import UnifiedInput
from .prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    content: str
    metadata: Dict[str, Any]
    processing_time: float
    token_usage: Dict[str, int]
    extra_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'content': self.content,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'token_usage': self.token_usage
        }
        if self.extra_data:
            result['extra_data'] = self.extra_data
        return result

class GraphProjector(nn.Module):
    """图嵌入投影器 - 基于G-Retriever设计"""
    
    def __init__(self, graph_dim: int = 128, llm_dim: int = 4096, hidden_dim: int = 512):
        """初始化投影器
        
        Args:
            graph_dim: 图嵌入维度（来自GraphEncoder）
            llm_dim: LLM词汇表空间维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.graph_dim = graph_dim
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim
        
        # 多层投影网络
        self.projection = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, llm_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            graph_embedding: 图嵌入张量 [batch_size, graph_dim]
            
        Returns:
            投影后的张量 [batch_size, llm_dim]
        """
        return self.projection(graph_embedding)

class LLMEngine:
    """LLM推理引擎"""
    
    def __init__(self, config: LLMConfig):
        """初始化LLM引擎"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.graph_projector = None
        self.prompt_manager = PromptTemplateManager()
        
        # 状态跟踪
        self.is_loaded = False
        self.device = config.model_config.device
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0,
            'multimodal_requests': 0,
            'text_only_requests': 0
        }
        
        logger.info(f"🤖 LLMEngine初始化完成，模型: {config.model_config.model_name}")
    
    def load_model(self) -> bool:
        """加载模型和tokenizer"""
        if not HAS_TRANSFORMERS:
            logger.error("❌ transformers库未安装，无法加载模型")
            return False
        
        try:
            start_time = time.time()
            model_config = self.config.model_config
            
            logger.info(f"🔄 开始加载模型: {model_config.model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True),
                padding_side='left'
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            model_kwargs = {
                'trust_remote_code': getattr(model_config, 'trust_remote_code', True),
                'torch_dtype': getattr(torch, model_config.torch_dtype, torch.float32),
                'device_map': 'auto' if model_config.device == 'cuda' else None,
                'attn_implementation': getattr(model_config, 'attn_implementation', 'eager')
            }
            
            # MacOS MPS支持
            if model_config.device == 'mps':
                model_kwargs['device_map'] = None
                model_kwargs['torch_dtype'] = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                **model_kwargs
            )
            
            # 移动到指定设备
            if model_config.device in ['cpu', 'mps']:
                self.model = self.model.to(model_config.device)
            
            self.model.eval()
            
            # 初始化图投影器
            if self.config.enable_multimodal:
                self._init_graph_projector()
            
            loading_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"✅ 模型加载成功，耗时: {loading_time:.2f}秒")
            logger.info(f"📊 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            self.is_loaded = False
            return False
    
    def _init_graph_projector(self):
        """初始化图投影器"""
        try:
            # 获取LLM的隐藏维度
            llm_hidden_dim = self.model.config.hidden_size
            
            self.graph_projector = GraphProjector(
                graph_dim=self.config.graph_embedding_dim,
                llm_dim=llm_hidden_dim,
                hidden_dim=512
            )
            
            self.graph_projector.to(self.device)
            self.graph_projector.eval()
            
            logger.info(f"🎯 图投影器初始化完成: {self.config.graph_embedding_dim}d → {llm_hidden_dim}d")
            
        except Exception as e:
            logger.error(f"❌ 图投影器初始化失败: {str(e)}")
            self.graph_projector = None
    
    def generate_response(self, unified_input: UnifiedInput) -> LLMResponse:
        """生成响应"""
        if not self.is_loaded:
            return self._create_error_response("模型未加载", unified_input)
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 更新统计
            if unified_input.has_multimodal_data():
                self.stats['multimodal_requests'] += 1
            else:
                self.stats['text_only_requests'] += 1
            
            # 构建prompt
            prompt = self._build_prompt(unified_input)
            
            # 处理多模态输入
            processed_input = self._process_multimodal_input(unified_input, prompt)
            
            # 生成响应
            response_text = self._generate_text(processed_input['text'], processed_input['extra_inputs'])
            
            # 后处理
            formatted_response = self._post_process_response(response_text, unified_input)
            
            processing_time = time.time() - start_time
            
            # 创建响应对象
            response = LLMResponse(
                content=formatted_response,
                metadata={
                    'processor_type': unified_input.processor_type,
                    'has_multimodal': unified_input.has_multimodal_data(),
                    'prompt_length': len(prompt),
                    'model_name': self.config.model_config.model_name,
                    'fusion_metadata': processed_input.get('fusion_metadata', {}),
                    'extra_inputs_provided': processed_input.get('extra_inputs') is not None
                },
                processing_time=processing_time,
                token_usage=self._calculate_token_usage(prompt, response_text),
                extra_data={
                    'fusion_strategy': self.config.fusion_strategy,
                    'processed_input': processed_input
                }
            )
            
            # 更新统计
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['total_tokens_generated'] += response.token_usage.get('output_tokens', 0)
            
            logger.info(f"✅ 响应生成成功，耗时: {processing_time:.2f}秒")
            return response
            
        except Exception as e:
            logger.error(f"❌ 响应生成失败: {str(e)}")
            self.stats['failed_requests'] += 1
            return self._create_error_response(str(e), unified_input)
    
    def _build_prompt(self, unified_input: UnifiedInput) -> str:
        """构建prompt"""
        try:
            system_prompt = self.config.system_prompt
            prompt = self.prompt_manager.format_prompt_for_input(unified_input, system_prompt)
            
            # 检查长度限制
            if len(prompt) > self.config.max_input_tokens * 4:  # 大概估算
                logger.warning(f"⚠️ Prompt过长，可能被截断")
            
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Prompt构建失败: {str(e)}")
            # 使用简单的回退prompt
            return f"<|user|>\n{unified_input.query}<|end|>\n<|assistant|>\n"
    
    def _process_multimodal_input(self, unified_input: UnifiedInput, prompt: str) -> Dict[str, Any]:
        """处理多模态输入 - 基于G-Retriever的深度融合"""
        result = {
            'text': prompt,
            'extra_inputs': None,
            'fusion_metadata': {}
        }
        
        # 如果没有多模态数据或投影器，直接返回文本
        if not unified_input.has_multimodal_data() or not self.graph_projector:
            result['fusion_metadata']['fusion_applied'] = False
            return result
        
        try:
            # 获取图嵌入
            graph_embedding = unified_input.get_graph_embedding()
            if not graph_embedding:
                result['fusion_metadata']['fusion_applied'] = False
                return result
            
            logger.info(f"🎭 开始多模态融合处理，图嵌入维度: {len(graph_embedding)}")
            
            # 转换为tensor
            graph_tensor = torch.tensor(graph_embedding, dtype=torch.float32).unsqueeze(0)  # [1, graph_dim]
            graph_tensor = graph_tensor.to(self.device)
            
            # 投影到LLM空间
            with torch.no_grad():
                projected_embedding = self.graph_projector(graph_tensor)  # [1, llm_dim]
            
            # 基于G-Retriever的融合策略实现
            fusion_result = self._apply_fusion_strategy(
                prompt, 
                projected_embedding, 
                unified_input, 
                graph_embedding
            )
            
            result.update(fusion_result)
            result['fusion_metadata'].update({
                'fusion_applied': True,
                'graph_embedding_dim': len(graph_embedding),
                'projected_dim': projected_embedding.shape[-1],
                'fusion_strategy': self.config.fusion_strategy,
                'has_multimodal_context': unified_input.multimodal_context is not None
            })
            
            logger.info(f"✅ 多模态融合完成，策略: {self.config.fusion_strategy}")
            
        except Exception as e:
            logger.error(f"❌ 多模态输入处理失败: {str(e)}")
            result['fusion_metadata']['fusion_applied'] = False
            result['fusion_metadata']['error'] = str(e)
        
        return result
    
    def _apply_fusion_strategy(self, prompt: str, projected_embedding: torch.Tensor, 
                             unified_input: UnifiedInput, graph_embedding: List[float]) -> Dict[str, Any]:
        """应用融合策略 - 基于G-Retriever论文实现"""
        try:
            if self.config.fusion_strategy == "concatenate":
                return self._concatenate_fusion(prompt, projected_embedding, unified_input, graph_embedding)
            elif self.config.fusion_strategy == "weighted":
                return self._weighted_fusion(prompt, projected_embedding, unified_input, graph_embedding)
            elif self.config.fusion_strategy == "attention":
                return self._attention_fusion(prompt, projected_embedding, unified_input, graph_embedding)
            else:
                # 默认concatenate
                return self._concatenate_fusion(prompt, projected_embedding, unified_input, graph_embedding)
                
        except Exception as e:
            logger.error(f"❌ 融合策略应用失败: {str(e)}")
            return {'text': prompt, 'extra_inputs': None}
    
    def _concatenate_fusion(self, prompt: str, projected_embedding: torch.Tensor, 
                          unified_input: UnifiedInput, graph_embedding: List[float]) -> Dict[str, Any]:
        """拼接融合策略 - G-Retriever核心方法"""
        # 1. 提取图谱语义信息描述
        graph_description = self._extract_graph_semantics(unified_input)
        
        # 2. 构建融合文本 - 将图谱信息嵌入到prompt中
        fusion_text = self._build_fusion_text(prompt, graph_description, unified_input)
        
        # 3. 准备额外输入（投影后的图嵌入用于潜在的深度融合）
        extra_inputs = {
            'graph_projection': projected_embedding,
            'raw_graph_embedding': graph_embedding,
            'graph_semantics': graph_description
        }
        
        return {
            'text': fusion_text,
            'extra_inputs': extra_inputs
        }
    
    def _weighted_fusion(self, prompt: str, projected_embedding: torch.Tensor, 
                        unified_input: UnifiedInput, graph_embedding: List[float]) -> Dict[str, Any]:
        """加权融合策略"""
        # 计算文本和图嵌入的权重
        text_weight = 0.7  # 文本权重
        graph_weight = 0.3  # 图权重
        
        # 构建加权描述
        graph_description = self._extract_graph_semantics(unified_input)
        weighted_description = f"(权重{graph_weight:.1f}) {graph_description}"
        
        fusion_text = self._build_fusion_text(prompt, weighted_description, unified_input)
        
        extra_inputs = {
            'graph_projection': projected_embedding * graph_weight,
            'weights': {'text': text_weight, 'graph': graph_weight},
            'fusion_mode': 'weighted'
        }
        
        return {
            'text': fusion_text,
            'extra_inputs': extra_inputs
        }
    
    def _attention_fusion(self, prompt: str, projected_embedding: torch.Tensor, 
                         unified_input: UnifiedInput, graph_embedding: List[float]) -> Dict[str, Any]:
        """注意力融合策略（简化版）"""
        # 计算注意力权重（简化实现）
        attention_score = min(1.0, len(graph_embedding) / 128.0)  # 基于嵌入维度的简单注意力
        
        graph_description = self._extract_graph_semantics(unified_input)
        attention_description = f"(注意力{attention_score:.2f}) {graph_description}"
        
        fusion_text = self._build_fusion_text(prompt, attention_description, unified_input)
        
        extra_inputs = {
            'graph_projection': projected_embedding * attention_score,
            'attention_score': attention_score,
            'fusion_mode': 'attention'
        }
        
        return {
            'text': fusion_text,
            'extra_inputs': extra_inputs
        }
    
    def _extract_graph_semantics(self, unified_input: UnifiedInput) -> str:
        """提取图谱语义信息"""
        try:
            # 从multimodal_context中提取语义信息
            if unified_input.multimodal_context:
                metadata = unified_input.multimodal_context.metadata
                if 'graph_summary' in metadata:
                    return metadata['graph_summary']
            
            # 从processor_data中提取图谱信息
            if unified_input.processor_data:
                if 'graph_embedding' in unified_input.processor_data:
                    graph_info = unified_input.processor_data['graph_embedding']
                    if isinstance(graph_info, dict) and 'encoding_metadata' in graph_info:
                        return graph_info['encoding_metadata'].get('summary', '图谱结构特征')
                
                # 从traditional_result中提取图谱结构信息
                if 'traditional_result' in unified_input.processor_data:
                    trad_result = unified_input.processor_data['traditional_result']
                    if 'graph' in trad_result:
                        graph_data = trad_result['graph']
                        return self._summarize_graph_structure(graph_data)
            
            # 默认描述
            graph_dim = len(unified_input.get_graph_embedding()) if unified_input.get_graph_embedding() else 0
            return f"图谱结构特征（{graph_dim}维向量表示）"
            
        except Exception as e:
            logger.warning(f"⚠️ 图谱语义提取失败: {str(e)}")
            return "图谱结构信息"
    
    def _summarize_graph_structure(self, graph_data: Dict[str, Any]) -> str:
        """总结图谱结构信息"""
        try:
            summary_parts = []
            
            if 'graph_structure' in graph_data:
                structure = graph_data['graph_structure']
                if 'num_nodes' in structure:
                    summary_parts.append(f"{structure['num_nodes']}个节点")
                if 'num_edges' in structure:
                    summary_parts.append(f"{structure['num_edges']}条边")
            
            if 'metadata' in graph_data:
                metadata = graph_data['metadata']
                if 'entity_types' in metadata:
                    types = metadata['entity_types']
                    if types:
                        summary_parts.append(f"实体类型: {', '.join(types[:3])}")
            
            return "图谱包含" + "、".join(summary_parts) if summary_parts else "图谱结构特征"
            
        except Exception:
            return "图谱结构特征"
    
    def _build_fusion_text(self, original_prompt: str, graph_description: str, 
                          unified_input: UnifiedInput) -> str:
        """构建融合文本 - 将图谱信息自然地融入prompt"""
        try:
            # 检查是否有文本上下文
            text_context = unified_input.get_text_content()
            
            # 构建图谱信息描述
            graph_info_block = f"""
[图谱分析]
{graph_description}
相关图谱上下文: {text_context[:200] + '...' if len(text_context) > 200 else text_context}
"""
            
            # 智能插入图谱信息
            if '<|assistant|>' in original_prompt:
                # 在assistant回复前插入图谱信息
                fusion_prompt = original_prompt.replace(
                    '<|assistant|>', 
                    f'{graph_info_block}\n<|assistant|>\n根据上述文本和图谱信息，我来回答：'
                )
            else:
                # 在prompt末尾添加图谱信息
                fusion_prompt = f"{original_prompt}\n\n{graph_info_block}\n\n请结合文本和图谱信息回答问题。"
            
            return fusion_prompt
            
        except Exception as e:
            logger.warning(f"⚠️ 融合文本构建失败: {str(e)}")
            return original_prompt
    
    def _generate_text(self, prompt: str, extra_inputs: Optional[Dict[str, Any]] = None) -> str:
        """生成文本"""
        try:
            # Tokenize输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_input_tokens,
                truncation=True,
                padding=False
            )
            
            # 移动到设备
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_output_tokens,
                temperature=self.config.model_config.temperature,
                top_p=self.config.model_config.top_p,
                top_k=self.config.model_config.top_k,
                do_sample=self.config.model_config.do_sample,
                num_beams=self.config.model_config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    use_cache=True
                )
            
            # 解码响应
            response_ids = outputs[0][input_ids.shape[1]:]  # 只取新生成的部分
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"❌ 文本生成失败: {str(e)}")
            return f"抱歉，生成响应时出现错误: {str(e)}"
    
    def _post_process_response(self, response_text: str, unified_input: UnifiedInput) -> str:
        """后处理响应"""
        try:
            # 基本清理
            response = response_text.strip()
            
            # 移除可能的重复或异常内容
            if response.count('\n') > 10:  # 过多换行
                lines = response.split('\n')
                response = '\n'.join(lines[:10])
            
            # 确保响应不为空
            if not response:
                response = "抱歉，我无法基于提供的信息回答这个问题。"
            
            # Markdown格式化（如果启用）
            if self.config.format_markdown:
                response = self._format_as_markdown(response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 响应后处理失败: {str(e)}")
            return response_text
    
    def _format_as_markdown(self, text: str) -> str:
        """格式化为Markdown"""
        # 简单的Markdown格式化
        formatted = text
        
        # 添加适当的间距
        if not formatted.endswith('\n'):
            formatted += '\n'
        
        return formatted
    
    def _calculate_token_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """计算token使用量"""
        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = len(self.tokenizer.encode(response))
            
            return {
                'input_tokens': prompt_tokens,
                'output_tokens': response_tokens,
                'total_tokens': prompt_tokens + response_tokens
            }
        except:
            return {
                'input_tokens': len(prompt) // 4,  # 粗略估算
                'output_tokens': len(response) // 4,
                'total_tokens': (len(prompt) + len(response)) // 4
            }
    
    def _create_error_response(self, error_message: str, unified_input: UnifiedInput) -> LLMResponse:
        """创建错误响应"""
        return LLMResponse(
            content=f"抱歉，处理您的请求时出现错误: {error_message}",
            metadata={
                'error': True,
                'error_message': error_message,
                'processor_type': unified_input.processor_type
            },
            processing_time=0.0,
            token_usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = max(1, self.stats['total_requests'])
        
        return {
            'model_loaded': self.is_loaded,
            'model_name': self.config.model_config.model_name,
            'device': self.device,
            'multimodal_enabled': self.config.enable_multimodal,
            'graph_projector_enabled': self.graph_projector is not None,
            'requests': {
                'total': self.stats['total_requests'],
                'successful': self.stats['successful_requests'],
                'failed': self.stats['failed_requests'],
                'success_rate': self.stats['successful_requests'] / total_requests
            },
            'performance': {
                'avg_processing_time': self.stats['total_processing_time'] / max(1, self.stats['successful_requests']),
                'total_tokens_generated': self.stats['total_tokens_generated'],
                'avg_tokens_per_request': self.stats['total_tokens_generated'] / max(1, self.stats['successful_requests'])
            },
            'input_distribution': {
                'multimodal_requests': self.stats['multimodal_requests'],
                'text_only_requests': self.stats['text_only_requests'],
                'multimodal_ratio': self.stats['multimodal_requests'] / total_requests
            }
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'total_processing_time': 0.0,
            'multimodal_requests': 0,
            'text_only_requests': 0
        }

# =============================================================================
# 工厂函数
# =============================================================================

def create_llm_engine(config: Optional[LLMConfig] = None) -> LLMEngine:
    """创建LLM引擎"""
    if config is None:
        from .config import get_macos_optimized_config
        config = get_macos_optimized_config()
    
    return LLMEngine(config)

# =============================================================================
# 测试和演示
# =============================================================================

if __name__ == "__main__":
    # 创建配置
    from .config import get_macos_optimized_config
    from .input_router import UnifiedInput
    
    print("🤖 LLM引擎测试")
    
    # 创建引擎
    config = get_macos_optimized_config()
    engine = create_llm_engine(config)
    
    print(f"引擎配置: {config.model_config.model_name}")
    print(f"设备: {config.model_config.device}")
    print(f"多模态支持: {config.enable_multimodal}")
    
    # 注意：实际加载模型需要较长时间，这里只做配置测试
    print("✅ 引擎创建成功")
    
    # 创建测试输入
    test_input = UnifiedInput(
        query="科比是谁？",
        processor_type="direct",
        text_context="科比是洛杉矶湖人队的传奇球员。"
    )
    
    print(f"测试输入: {test_input}")
    print(f"统计信息: {engine.get_stats()}")
