"""
Core module: Unified data objects and interface specifications
Provides standardized data structures and contracts for the entire system
"""

from .schemas import (
    # 基础数据结构
    LanguageInfo,
    EntityInfo, 
    IntentInfo,
    RAGResult,
    LLMResult,
    
    # 核心上下文对象
    QueryContext,
    
    # 工厂类
    QueryContextFactory
)

__all__ = [
    'LanguageInfo',
    'EntityInfo',
    'IntentInfo', 
    'RAGResult',
    'LLMResult',
    'QueryContext',
    'QueryContextFactory'
]
