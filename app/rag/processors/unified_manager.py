"""
Unified Processor Manager
Specifically manages next-generation processors that support QueryContext
"""
import logging
from typing import Dict, Any, Optional

from app.core.schemas import QueryContext, RAGResult
from app.rag.processors.context_aware_processor import (
    ContextAwareProcessor, 
    DirectContextProcessor, 
    ChitchatContextProcessor
)

logger = logging.getLogger(__name__)

class UnifiedProcessorManager:
    """统一处理器管理器 - 管理支持QueryContext的处理器"""
    
    def __init__(self):
        """初始化统一处理器管理器"""
        self._active_processors: Dict[str, ContextAwareProcessor] = {}
        
        # 处理器映射
        self._processor_classes = {
            'direct_db_processor': DirectContextProcessor,
            'chitchat_processor': ChitchatContextProcessor,
            # 未来可以扩展更多处理器
        }
        
        logger.info("🔄 统一处理器管理器初始化完成")
    
    def get_processor(self, processor_name: str, 
                     config: Optional[Dict[str, Any]] = None) -> ContextAwareProcessor:
        """获取处理器实例（带缓存）"""
        
        cache_key = f"{processor_name}_{hash(str(config))}"
        
        if cache_key not in self._active_processors:
            # 创建新处理器
            if processor_name not in self._processor_classes:
                available_processors = list(self._processor_classes.keys())
                raise ValueError(f"未支持的处理器: {processor_name}。可用: {available_processors}")
            
            processor_class = self._processor_classes[processor_name]
            processor = processor_class(config)
            
            # 初始化处理器
            if not processor.initialize():
                raise RuntimeError(f"处理器 {processor_name} 初始化失败")
            
            self._active_processors[cache_key] = processor
            logger.info(f"📁 缓存新处理器: {processor_name}")
        
        return self._active_processors[cache_key]
    
    def process_with_context(self, processor_name: str, context: QueryContext,
                           config: Optional[Dict[str, Any]] = None) -> RAGResult:
        """使用指定处理器处理QueryContext"""
        try:
            processor = self.get_processor(processor_name, config)
            return processor.process_query(context)
        except Exception as e:
            logger.error(f"❌ 处理器 {processor_name} 处理失败: {str(e)}")
            # 返回错误结果
            return RAGResult(
                success=False,
                processor_used=processor_name,
                error_message=f"处理器执行失败: {str(e)}",
                context_text="",
                retrieved_nodes=[],
                confidence=0.0,
                processing_strategy=f"{processor_name}_error"
            )
    
    def list_available_processors(self) -> Dict[str, str]:
        """列出所有可用的处理器"""
        return {
            'direct_db_processor': '直接数据库查询处理器',
            'chitchat_processor': '闲聊处理器',
        }
    
    def get_processor_stats(self, processor_name: str) -> Optional[Dict[str, Any]]:
        """获取处理器统计信息"""
        for cache_key, processor in self._active_processors.items():
            if processor.processor_name == processor_name:
                return processor.get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有活跃处理器的统计信息"""
        stats = {}
        for cache_key, processor in self._active_processors.items():
            stats[processor.processor_name] = processor.get_stats()
        return stats
    
    def reset_all_stats(self):
        """重置所有处理器的统计信息"""
        for processor in self._active_processors.values():
            processor.reset_stats()
        logger.info("📊 所有处理器统计信息已重置")
    
    def shutdown(self):
        """关闭管理器"""
        self._active_processors.clear()
        logger.info("🔄 统一处理器管理器已关闭")

# 全局实例
unified_processor_manager = UnifiedProcessorManager()
