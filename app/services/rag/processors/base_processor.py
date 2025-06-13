"""
RAG 处理器基类
定义所有处理器的通用接口和功能
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging
from app.rag.cache_manager import CacheManager
from app.rag.components import component_factory, ProcessorConfig

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """RAG处理器基类"""
    
    def __init__(self, config: ProcessorConfig):
        """初始化处理器"""
        self.config = config
        self.processor_name = config.processor_name
        
        # 组件实例
        self.retriever = None
        self.graph_builder = None
        self.textualizer = None
        
        # 缓存管理器
        self.cache_manager = None
        if config.cache_enabled:
            self.cache_manager = CacheManager({
                'memory_size': 1000,
                'disk_cache_dir': 'cache',
                'default_ttl': config.cache_ttl
            })
        
        self.is_initialized = False
        self.stats = {
            'queries_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'errors': 0
        }
    
    def initialize(self) -> bool:
        """初始化处理器"""
        try:
            logger.info(f"🔄 初始化处理器: {self.processor_name}")
            
            # 创建组件
            components = component_factory.create_processor_components(self.config)
            
            self.retriever = components['retriever']
            self.graph_builder = components['graph_builder']
            self.textualizer = components['textualizer']
            
            # 初始化缓存管理器
            if self.cache_manager:
                # CacheManager不需要单独的initialize方法，在构造时已经初始化
                logger.info(f"✅ 缓存管理器已就绪，处理器 {self.processor_name} 将使用缓存")
            else:
                logger.info(f"📝 处理器 {self.processor_name} 未启用缓存")
            
            self.is_initialized = True
            logger.info(f"✅ 处理器 {self.processor_name} 初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 处理器 {self.processor_name} 初始化失败: {str(e)}")
            return False
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理查询"""
        if not self.is_initialized:
            raise RuntimeError(f"处理器 {self.processor_name} 未初始化")
        
        start_time = time.time()
        
        try:
            # 构建缓存键
            cache_key = None
            if self.cache_manager:
                cache_key = self._build_cache_key(query, context)
                
                # 尝试从缓存获取
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.stats['cache_hits'] += 1
                    self.stats['queries_processed'] += 1
                    
                    logger.info(f"🎯 缓存命中: {self.processor_name}")
                    return self._add_metadata(cached_result, from_cache=True)
            
            # 缓存未命中，执行处理
            if self.cache_manager:
                self.stats['cache_misses'] += 1
            
            # 调用具体处理逻辑
            result = self._process_impl(query, context)
            
            # 缓存结果
            if self.cache_manager and cache_key:
                self.cache_manager.set(cache_key, result)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return self._add_metadata(result, processing_time=processing_time)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ 处理器 {self.processor_name} 处理失败: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'processor': self.processor_name,
                'query': query,
                'context': context or {},
                'timestamp': time.time()
            }
    
    @abstractmethod
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """具体处理逻辑（由子类实现）"""
        pass
    
    def _build_cache_key(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """构建缓存键"""
        context_str = str(sorted(context.items())) if context else ""
        return f"{query}|{context_str}"
    
    def _add_metadata(self, result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """添加元数据"""
        if isinstance(result, dict):
            result = result.copy()
            result.update({
                'processor': self.processor_name,
                'timestamp': time.time(),
                **kwargs
            })
        return result
    
    def _update_stats(self, processing_time: float):
        """更新统计信息"""
        self.stats['queries_processed'] += 1
        
        # 更新平均处理时间
        total_queries = self.stats['queries_processed']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        cache_stats = {}
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
        
        return {
            'processor_name': self.processor_name,
            'is_initialized': self.is_initialized,
            'processing_stats': self.stats.copy(),
            'cache_stats': cache_stats,
            'config': {
                'cache_enabled': self.config.cache_enabled,
                'cache_ttl': self.config.cache_ttl,
                'max_tokens': self.config.max_tokens
            }
        }
    
    def clear_cache(self) -> bool:
        """清空缓存"""
        if self.cache_manager:
            return self.cache_manager.clear()
        return True
    
    def get_component_info(self) -> Dict[str, str]:
        """获取组件信息"""
        return {
            'retriever': self.config.retriever_config.component_name,
            'graph_builder': self.config.graph_builder_config.component_name,
            'textualizer': self.config.textualizer_config.component_name
        }

# =============================================================================
# 通用处理流程辅助方法
# =============================================================================

class ProcessorUtils:
    """处理器工具类"""
    
    @staticmethod
    def extract_entities_from_query(query: str) -> Dict[str, Any]:
        """从查询中提取实体信息"""
        entities = {
            'players': [],
            'teams': [],
            'attributes': [],
            'numbers': []
        }
        
        query_lower = query.lower()
        
        # 简单的实体提取逻辑
        import re
        
        # 提取数字
        numbers = re.findall(r'\d+', query)
        entities['numbers'] = numbers
        
        # 扩展的球员名称库（支持中英文）
        player_names = [
            # 英文名称
            'yao ming', 'kobe bryant', 'lebron james', 'michael jordan', 'stephen curry',
            'kevin durant', 'james harden', 'russell westbrook', 'chris paul', 'carmelo anthony',
            'tracy mcgrady', 'dwight howard', 'shaquille oneal', 'tim duncan', 'magic johnson',
            'larry bird', 'kareem abdul-jabbar', 'wilt chamberlain', 'bill russell',
            # 简短版本
            'yao', 'kobe', 'lebron', 'jordan', 'curry', 'durant', 'harden', 'westbrook',
            'paul', 'anthony', 'mcgrady', 'howard', 'shaq', 'duncan', 'magic',
            # 中文名称
            '姚明', '科比', '詹姆斯', '乔丹', '库里', '杜兰特', '哈登', '威少'
        ]
        
        # 扩展的球队名称库（支持中英文）
        team_names = [
            # 英文名称
            'lakers', 'warriors', 'bulls', 'celtics', 'heat', 'spurs', 'rockets', 'nets',
            'thunder', 'clippers', 'mavericks', 'knicks', 'hawks', 'pacers', 'cavaliers',
            'los angeles lakers', 'golden state warriors', 'chicago bulls', 'boston celtics',
            'miami heat', 'san antonio spurs', 'houston rockets', 'brooklyn nets',
            # 中文名称
            '湖人', '勇士', '公牛', '凯尔特人', '热火', '马刺', '火箭', '篮网', '雷霆'
        ]
        
        # 检查球员名称
        for player in player_names:
            if player in query_lower:
                entities['players'].append(player)
        
        # 检查球队名称
        for team in team_names:
            if team in query_lower:
                entities['teams'].append(team)
        
        # 扩展的属性关键词（支持中英文）
        attribute_keywords = [
            # 中文
            '年龄', '身高', '体重', '得分', '助攻', '篮板', '位置', '球衣号码',
            # 英文
            'age', 'old', 'height', 'tall', 'weight', 'position', 'jersey', 'number',
            'stats', 'points', 'assists', 'rebounds', 'born', 'birthday'
        ]
        
        for attr in attribute_keywords:
            if attr in query_lower:
                entities['attributes'].append(attr)
        
        return entities
    
    @staticmethod
    def validate_subgraph(subgraph: Dict[str, Any]) -> bool:
        """验证子图结构"""
        if not isinstance(subgraph, dict):
            return False
        
        required_keys = ['nodes', 'edges', 'node_count', 'edge_count']
        for key in required_keys:
            if key not in subgraph:
                return False
        
        # 验证节点和边是否为列表
        if not isinstance(subgraph['nodes'], list):
            return False
        if not isinstance(subgraph['edges'], list):
            return False
        
        return True
    
    @staticmethod
    def limit_tokens(text: str, max_tokens: int) -> str:
        """限制文本长度"""
        if len(text) <= max_tokens:
            return text
        
        # 简单截断并添加省略号
        truncated = text[:max_tokens - 3]
        
        # 尝试在句子边界截断
        last_period = truncated.rfind('。')
        last_newline = truncated.rfind('\n')
        
        cut_point = max(last_period, last_newline)
        if cut_point > max_tokens * 0.8:  # 如果截断点不会损失太多内容
            truncated = truncated[:cut_point + 1]
        
        return truncated + "..."
    
    @staticmethod
    def merge_results(results: list, max_items: int = 10) -> list:
        """合并多个结果列表"""
        if not results:
            return []
        
        # 去重并合并
        seen = set()
        merged = []
        
        for result_list in results:
            if not isinstance(result_list, list):
                continue
                
            for item in result_list:
                if isinstance(item, dict):
                    # 使用node_id或id作为去重键
                    key = item.get('node_id') or item.get('id') or str(item)
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)
                        
                        if len(merged) >= max_items:
                            return merged
        
        return merged
