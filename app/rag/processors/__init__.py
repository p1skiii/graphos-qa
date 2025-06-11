"""
RAG 处理器模块初始化
提供统一的处理器接口和工厂方法
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 导入所有处理器
from .base_processor import BaseProcessor, ProcessorUtils
from .direct_processor import DirectProcessor, create_direct_processor
from .simple_g_processor import SimpleGProcessor, create_simple_g_processor
from .complex_g_processor import ComplexGProcessor, create_complex_g_processor
from .comparison_processor import ComparisonProcessor, create_comparison_processor
from .chitchat_processor import ChitchatProcessor, create_chitchat_processor
from .gnn_processor import GNNProcessor, create_gnn_processor

# =============================================================================
# 处理器工厂
# =============================================================================

class ProcessorFactory:
    """处理器工厂类"""
    
    def __init__(self):
        """初始化处理器工厂"""
        self._processor_registry = {
            'direct': DirectProcessor,
            'simple_g': SimpleGProcessor,
            'complex_g': ComplexGProcessor,
            'comparison': ComparisonProcessor,
            'chitchat': ChitchatProcessor,
            'gnn': GNNProcessor
        }
        
        self._create_functions = {
            'direct': create_direct_processor,
            'simple_g': create_simple_g_processor,
            'complex_g': create_complex_g_processor,
            'comparison': create_comparison_processor,
            'chitchat': create_chitchat_processor,
            'gnn': create_gnn_processor
        }
    
    def create_processor(self, processor_type: str, 
                        custom_config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
        """创建处理器实例"""
        if processor_type not in self._create_functions:
            available_types = list(self._create_functions.keys())
            raise ValueError(f"未支持的处理器类型: {processor_type}。可用类型: {available_types}")
        
        create_func = self._create_functions[processor_type]
        processor = create_func(custom_config)
        
        logger.info(f"✅ 创建处理器: {processor_type}")
        return processor
    
    def list_available_processors(self) -> list:
        """列出所有可用的处理器类型"""
        return list(self._processor_registry.keys())
    
    def get_processor_info(self, processor_type: str) -> Dict[str, Any]:
        """获取处理器信息"""
        processor_info = {
            'direct': {
                'name': '直接查询处理器',
                'description': '处理直接的属性查询，如年龄、身高等',
                'best_for': ['属性查询', '事实查询', '简单问答'],
                'example_queries': ['科比多少岁', '湖人队主场在哪里']
            },
            'simple_g': {
                'name': '简单图查询处理器',
                'description': '处理简单的关系查询，如效力关系',
                'best_for': ['关系查询', '一跳连接', '直接关联'],
                'example_queries': ['科比在哪个球队', '湖人队有哪些球员']
            },
            'complex_g': {
                'name': '复杂图查询处理器',
                'description': '处理复杂的多跳关系查询',
                'best_for': ['多跳关系', '路径查找', '复杂关联'],
                'example_queries': ['科比和詹姆斯有什么共同点', '通过什么关系连接']
            },
            'comparison': {
                'name': '比较查询处理器',
                'description': '处理比较类查询',
                'best_for': ['实体比较', '优劣对比', '相似性分析'],
                'example_queries': ['科比和詹姆斯谁更强', '湖人和勇士哪个队更好']
            },
            'chitchat': {
                'name': '闲聊处理器',
                'description': '处理篮球领域的闲聊查询',
                'best_for': ['话题讨论', '观点交流', '推荐咨询'],
                'example_queries': ['聊聊篮球', 'NBA怎么样', '你觉得哪个球员最厉害']
            },
            'gnn': {
                'name': 'GNN处理器',
                'description': '使用图神经网络处理复杂图查询',
                'best_for': ['图学习', '节点分类', '图嵌入'],
                'example_queries': ['分析球员关系网络', '预测球队表现', '发现潜在连接']
            }
        }
        
        return processor_info.get(processor_type, {})

# =============================================================================
# 全局工厂实例
# =============================================================================

processor_factory = ProcessorFactory()

# =============================================================================
# 处理器管理器
# =============================================================================

class ProcessorManager:
    """处理器管理器"""
    
    def __init__(self):
        """初始化处理器管理器"""
        self._active_processors: Dict[str, BaseProcessor] = {}
        self._processor_factory = processor_factory
    
    def get_processor(self, processor_type: str, 
                     custom_config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
        """获取处理器实例（带缓存）"""
        cache_key = f"{processor_type}_{hash(str(custom_config))}"
        
        if cache_key not in self._active_processors:
            processor = self._processor_factory.create_processor(processor_type, custom_config)
            
            # 初始化处理器
            if not processor.initialize():
                raise RuntimeError(f"处理器 {processor_type} 初始化失败")
            
            self._active_processors[cache_key] = processor
            logger.info(f"📁 缓存处理器: {processor_type}")
        
        return self._active_processors[cache_key]
    
    def clear_processor_cache(self):
        """清空处理器缓存"""
        for processor in self._active_processors.values():
            try:
                processor.clear_cache()
            except:
                pass
        
        self._active_processors.clear()
        logger.info("🗑️ 处理器缓存已清空")
    
    def get_all_processor_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有活跃处理器的统计信息"""
        stats = {}
        for cache_key, processor in self._active_processors.items():
            stats[cache_key] = processor.get_stats()
        return stats
    
    def shutdown(self):
        """关闭管理器"""
        self.clear_processor_cache()
        logger.info("🔄 处理器管理器已关闭")

# =============================================================================
# 全局管理器实例
# =============================================================================

processor_manager = ProcessorManager()

# =============================================================================
# 便捷函数
# =============================================================================

def create_processor(processor_type: str, 
                    custom_config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """便捷的处理器创建函数"""
    return processor_factory.create_processor(processor_type, custom_config)

def get_processor(processor_type: str, 
                 custom_config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """便捷的处理器获取函数（带缓存）"""
    return processor_manager.get_processor(processor_type, custom_config)

def process_query(processor_type: str, query: str, 
                 context: Optional[Dict[str, Any]] = None,
                 custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """便捷的查询处理函数"""
    processor = get_processor(processor_type, custom_config)
    return processor.process(query, context)

# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    # 基础类
    'BaseProcessor',
    'ProcessorUtils',
    
    # 具体处理器
    'DirectProcessor',
    'SimpleGProcessor', 
    'ComplexGProcessor',
    'ComparisonProcessor',
    'ChitchatProcessor',
    'GNNProcessor',
    
    # 工厂和管理器
    'ProcessorFactory',
    'ProcessorManager',
    'processor_factory',
    'processor_manager',
    
    # 便捷函数
    'create_processor',
    'get_processor',
    'process_query',
    
    # 创建函数
    'create_direct_processor',
    'create_simple_g_processor',
    'create_complex_g_processor',
    'create_comparison_processor',
    'create_chitchat_processor',
    'create_gnn_processor'
    'create_comparison_processor',
    'create_chitchat_processor'
]
