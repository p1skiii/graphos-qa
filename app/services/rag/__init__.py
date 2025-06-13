"""
RAG模块初始化 - 新架构
基于组件工厂模式的模块化RAG系统
"""

# 导入缓存管理器
from .cache_manager import CacheManager, cache_manager

# 导入组件工厂
from .component_factory import (
    ComponentFactory, 
    component_factory,
    BaseRetriever, 
    BaseGraphBuilder, 
    BaseTextualizer,
    ComponentConfig,
    ProcessorConfig,
    DefaultConfigs,
    ProcessorDefaultConfigs
)

# 导入所有组件（这会自动注册到工厂）
from .components import *

# 导入处理器（将在创建后导入）
# from .processors import *

__all__ = [
    # 缓存管理器
    'CacheManager', 'cache_manager',
    
    # 组件工厂
    'ComponentFactory', 'component_factory',
    'BaseRetriever', 'BaseGraphBuilder', 'BaseTextualizer',
    'ComponentConfig', 'ProcessorConfig', 
    'DefaultConfigs', 'ProcessorDefaultConfigs',
]
