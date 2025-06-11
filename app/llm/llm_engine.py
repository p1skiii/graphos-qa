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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'token_usage': self.token_usage
        }

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
                    'model_name': self.config.model_config.model_name
                },
                processing_time=processing_time,
                token_usage=self._calculate_token_usage(prompt, response_text)
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
        """处理多模态输入"""
        result = {
            'text': prompt,
            'extra_inputs': None
        }
        
        # 如果没有多模态数据或投影器，直接返回文本
        if not unified_input.has_multimodal_data() or not self.graph_projector:
            return result
        
        try:
            # 获取图嵌入
            graph_embedding = unified_input.get_graph_embedding()
            if not graph_embedding:
                return result
            
            # 转换为tensor
            graph_tensor = torch.tensor(graph_embedding, dtype=torch.float32).unsqueeze(0)  # [1, graph_dim]
            graph_tensor = graph_tensor.to(self.device)
            
            # 投影到LLM空间
            with torch.no_grad():
                projected_embedding = self.graph_projector(graph_tensor)  # [1, llm_dim]
            
            # 基于G-Retriever的融合策略
            if self.config.fusion_strategy == "concatenate":
                # 在这里我们暂时将图信息编码到文本中
                # 实际的concatenate融合需要在模型内部实现
                graph_info = f"\n[图谱特征: {len(graph_embedding)}维向量表示]\n"
                result['text'] = prompt.replace('<|assistant|>', graph_info + '<|assistant|>')
                result['extra_inputs'] = {'graph_projection': projected_embedding}
            
            logger.info(f"🎭 多模态输入处理完成，图嵌入维度: {len(graph_embedding)}")
            
        except Exception as e:
            logger.error(f"❌ 多模态输入处理失败: {str(e)}")
        
        return result
    
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
