"""
LLM模块初始化
基于G-Retriever论文的三阶段LLM集成方案
支持Phi-3-mini模型和多模态输入处理
"""

# 配置模块
from .config import LLMConfig, Phi3Config, ModelConfig

# 输入处理
from .input_router import InputRouter, UnifiedInput, LLMInputConfig, create_input_router

# Prompt管理
from .prompt_templates import PromptTemplateManager, PromptTemplate, create_prompt_template_manager

# 响应格式化
from .response_formatter import ResponseFormatter, create_response_formatter

# 核心引擎
from .llm_engine import LLMEngine, LLMResponse, create_llm_engine

# 工厂
from .factory import LLMFactory, LLMSystem, llm_factory, create_llm_system

__all__ = [
    # 核心引擎
    'LLMEngine',
    'create_llm_engine',
    
    # 输入处理
    'InputRouter', 
    'UnifiedInput',
    'LLMInputConfig',
    
    # Prompt管理
    'PromptTemplateManager',
    'PromptTemplate',
    
    # 响应格式化
    'ResponseFormatter',
    
    # 配置
    'LLMConfig',
    'Phi3Config', 
    'ModelConfig',
    
    # 工厂
    'LLMFactory',
    'LLMSystem',
    'llm_factory',
    'create_llm_system'
]
