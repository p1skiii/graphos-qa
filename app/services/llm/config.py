"""
LLM配置模块
定义Phi-3-mini和其他LLM模型的配置
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

@dataclass
class ModelConfig:
    """基础模型配置"""
    model_name: str
    model_path: Optional[str] = None
    device: str = "cpu"  # MacOS环境优先使用CPU
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 内存优化配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float32"
    
    # 推理配置
    batch_size: int = 1
    num_beams: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'max_length': self.max_length,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'do_sample': self.do_sample,
            'load_in_8bit': self.load_in_8bit,
            'load_in_4bit': self.load_in_4bit,
            'torch_dtype': self.torch_dtype,
            'batch_size': self.batch_size,
            'num_beams': self.num_beams
        }

@dataclass
class Phi3Config(ModelConfig):
    """Phi-3-mini专用配置"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    model_path: Optional[str] = None
    device: str = "cpu"  # MacOS环境使用CPU
    max_length: int = 4096
    temperature: float = 0.7
    
    # Phi-3特有配置
    trust_remote_code: bool = True
    attn_implementation: str = "eager"  # 兼容性更好
    
    # 提示词格式
    system_prompt_template: str = "<|system|>\n{system_message}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    user_prompt_template: str = "<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base_dict = super().to_dict()
        base_dict.update({
            'trust_remote_code': self.trust_remote_code,
            'attn_implementation': self.attn_implementation,
            'system_prompt_template': self.system_prompt_template,
            'user_prompt_template': self.user_prompt_template
        })
        return base_dict

@dataclass 
class LLMConfig:
    """LLM系统总配置"""
    # 模型配置
    model_config: ModelConfig = field(default_factory=lambda: Phi3Config())
    
    # 输入处理配置
    enable_multimodal: bool = True
    max_input_tokens: int = 3072  # 为输出预留1024 tokens
    graph_embedding_dim: int = 128
    text_embedding_dim: int = 768
    
    # 模态融合配置
    fusion_strategy: str = "concatenate"  # concatenate, weighted, attention
    graph_projection_dim: int = 4096  # 投影到LLM词汇表空间
    
    # Prompt配置
    use_system_prompt: bool = True
    system_prompt: str = "你是一个专业的篮球知识问答助手，基于提供的图谱信息和文本上下文回答用户问题。"
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1小时
    
    # 处理器映射
    processor_mapping: Dict[str, str] = field(default_factory=lambda: {
        'direct': 'simple_qa',
        'simple_g': 'simple_qa', 
        'complex_g': 'multimodal_qa',
        'comparison': 'comparison_qa',
        'chitchat': 'chitchat'
    })
    
    # 输出配置
    max_output_tokens: int = 1024
    include_metadata: bool = True
    format_markdown: bool = True
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.model_config
    
    def get_phi3_config(self) -> Phi3Config:
        """获取Phi-3配置（如果适用）"""
        if isinstance(self.model_config, Phi3Config):
            return self.model_config
        else:
            # 创建默认Phi-3配置
            return Phi3Config()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'model_config': self.model_config.to_dict(),
            'enable_multimodal': self.enable_multimodal,
            'max_input_tokens': self.max_input_tokens,
            'graph_embedding_dim': self.graph_embedding_dim,
            'text_embedding_dim': self.text_embedding_dim,
            'fusion_strategy': self.fusion_strategy,
            'graph_projection_dim': self.graph_projection_dim,
            'use_system_prompt': self.use_system_prompt,
            'system_prompt': self.system_prompt,
            'enable_cache': self.enable_cache,
            'cache_ttl': self.cache_ttl,
            'processor_mapping': self.processor_mapping,
            'max_output_tokens': self.max_output_tokens,
            'include_metadata': self.include_metadata,
            'format_markdown': self.format_markdown
        }

# =============================================================================
# 默认配置
# =============================================================================

def get_default_llm_config() -> LLMConfig:
    """获取默认LLM配置"""
    return LLMConfig()

def get_phi3_config() -> Phi3Config:
    """获取默认Phi-3配置"""
    return Phi3Config()

def get_macos_optimized_config() -> LLMConfig:
    """获取MacOS优化配置"""
    phi3_config = Phi3Config(
        device="cpu",  # MacOS环境使用CPU
        load_in_8bit=False,  # CPU不支持量化
        torch_dtype="float32",  # CPU使用float32
        batch_size=1,  # 单批次处理
        max_length=4096
    )
    
    return LLMConfig(
        model_config=phi3_config,
        enable_multimodal=True,
        max_input_tokens=3072,
        system_prompt="你是一个专业的篮球知识问答助手。基于提供的图谱信息和文本上下文，准确回答用户的篮球相关问题。",
        enable_cache=True
    )

# =============================================================================
# 配置验证
# =============================================================================

def validate_config(config: LLMConfig) -> bool:
    """验证配置有效性"""
    try:
        # 检查模型配置
        if not config.model_config.model_name:
            raise ValueError("model_name不能为空")
        
        # 检查token限制
        if config.max_input_tokens + config.max_output_tokens > config.model_config.max_length:
            raise ValueError(f"输入+输出token数({config.max_input_tokens + config.max_output_tokens})超过模型最大长度({config.model_config.max_length})")
        
        # 检查融合策略
        valid_strategies = ["concatenate", "weighted", "attention"]
        if config.fusion_strategy not in valid_strategies:
            raise ValueError(f"不支持的融合策略: {config.fusion_strategy}")
        
        # 检查设备配置
        if config.model_config.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"不支持的设备类型: {config.model_config.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试配置
    config = get_macos_optimized_config()
    print("✅ MacOS优化配置创建成功")
    print(f"模型: {config.model_config.model_name}")
    print(f"设备: {config.model_config.device}")
    print(f"多模态: {config.enable_multimodal}")
    
    # 验证配置
    is_valid = validate_config(config)
    print(f"配置有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
