"""
支持QueryContext的RAG处理器基类
为统一数据流架构设计的新一代处理器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import logging

from app.core.schemas import QueryContext, RAGResult
from app.core.validation import global_monitor

logger = logging.getLogger(__name__)

class ContextAwareProcessor(ABC):
    """支持QueryContext的RAG处理器基类"""
    
    def __init__(self, processor_name: str, config: Optional[Dict[str, Any]] = None):
        """初始化上下文感知处理器"""
        self.processor_name = processor_name
        self.config = config or {}
        
        # 统计信息
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time': 0.0,
            'context_validations_passed': 0,
            'context_validations_failed': 0
        }
        
        # 初始化标志
        self.is_initialized = False
        
        logger.info(f"🔄 初始化上下文感知处理器: {self.processor_name}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化处理器"""
        pass
    
    @abstractmethod
    def process_with_context(self, context: QueryContext) -> RAGResult:
        """
        使用QueryContext处理查询
        
        Args:
            context: 查询上下文对象
            
        Returns:
            RAGResult: RAG处理结果
        """
        pass
    
    def process_query(self, context: QueryContext) -> RAGResult:
        """
        统一的查询处理入口
        包含验证、监控、错误处理等通用逻辑
        """
        start_time = time.time()
        
        try:
            # 添加处理开始的追踪
            context.add_trace(
                self.processor_name, 
                "started", 
                {
                    "processor_type": self.processor_name,
                    "config": self.config
                }
            )
            
            # 检查处理器是否已初始化
            if not self.is_initialized:
                context.add_warning(self.processor_name, "处理器未初始化，尝试初始化")
                if not self.initialize():
                    error_msg = f"处理器 {self.processor_name} 初始化失败"
                    context.add_error(self.processor_name, error_msg)
                    return self._create_error_result(error_msg)
            
            # 预处理验证
            validation_result = self._validate_context(context)
            if not validation_result['valid']:
                error_msg = f"上下文验证失败: {validation_result['errors']}"
                context.add_error(self.processor_name, error_msg)
                return self._create_error_result(error_msg)
            
            # 执行具体的处理逻辑
            result = self.process_with_context(context)
            
            # 后处理验证
            if not self._validate_result(result):
                context.add_warning(self.processor_name, "结果验证失败，但继续处理")
            
            # 添加成功追踪
            processing_time = time.time() - start_time
            context.add_trace(
                self.processor_name, 
                "completed", 
                {
                    "processing_time": processing_time,
                    "success": result.success,
                    "nodes_retrieved": len(result.retrieved_nodes) if result.retrieved_nodes else 0
                }
            )
            
            # 更新统计
            self._update_stats(processing_time, True)
            
            # 记录到全局监控
            global_monitor.record_processing_time(self.processor_name, processing_time)
            global_monitor.increment_success_count(self.processor_name)
            
            return result
            
        except Exception as e:
            # 处理异常
            processing_time = time.time() - start_time
            error_msg = f"处理器 {self.processor_name} 执行失败: {str(e)}"
            
            context.add_error(self.processor_name, error_msg)
            context.add_trace(
                self.processor_name, 
                "failed", 
                {
                    "processing_time": processing_time,
                    "error": str(e)
                }
            )
            
            # 更新统计
            self._update_stats(processing_time, False)
            
            # 记录到全局监控
            global_monitor.record_processing_time(self.processor_name, processing_time)
            global_monitor.increment_error_count(self.processor_name)
            
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def _validate_context(self, context: QueryContext) -> Dict[str, Any]:
        """验证查询上下文"""
        errors = []
        warnings = []
        
        # 基础验证
        if not context.original_query or not context.original_query.strip():
            errors.append("原始查询为空")
        
        if not context.request_id:
            errors.append("请求ID缺失")
        
        # 意图验证（如果需要）
        if self._requires_intent() and not context.intent_info:
            warnings.append("意图信息缺失")
        
        # 实体验证（如果需要）
        if self._requires_entities() and not context.entity_info:
            warnings.append("实体信息缺失")
        
        # 语言验证
        if not context.language_info:
            warnings.append("语言信息缺失")
        
        # 记录验证结果
        if errors:
            self.stats['context_validations_failed'] += 1
        else:
            self.stats['context_validations_passed'] += 1
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_result(self, result: RAGResult) -> bool:
        """验证处理结果"""
        if not isinstance(result, RAGResult):
            return False
        
        # 检查必要字段
        if not hasattr(result, 'success') or not hasattr(result, 'processor_used'):
            return False
        
        # 如果处理成功，检查结果内容
        if result.success:
            if not result.context_text and not result.retrieved_nodes:
                return False
        
        return True
    
    def _create_error_result(self, error_message: str) -> RAGResult:
        """创建错误结果"""
        return RAGResult(
            success=False,
            processor_used=self.processor_name,
            error_message=error_message,
            context_text="",
            retrieved_nodes=[],
            confidence=0.0,
            processing_strategy=f"{self.processor_name}_error"
        )
    
    def _update_stats(self, processing_time: float, success: bool):
        """更新统计信息"""
        self.stats['queries_processed'] += 1
        
        if success:
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1
        
        # 更新平均处理时间
        total_time = self.stats['avg_processing_time'] * (self.stats['queries_processed'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['queries_processed']
    
    def _requires_intent(self) -> bool:
        """是否需要意图信息（子类可重写）"""
        return False
    
    def _requires_entities(self) -> bool:
        """是否需要实体信息（子类可重写）"""
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            'processor_name': self.processor_name,
            'is_initialized': self.is_initialized,
            'stats': self.stats.copy(),
            'success_rate': (
                self.stats['successful_queries'] / self.stats['queries_processed'] 
                if self.stats['queries_processed'] > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time': 0.0,
            'context_validations_passed': 0,
            'context_validations_failed': 0
        }
        logger.info(f"📊 处理器 {self.processor_name} 统计信息已重置")

class DirectContextProcessor(ContextAwareProcessor):
    """直接数据库查询处理器 - QueryContext版本"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("direct_db_processor", config)
    
    def initialize(self) -> bool:
        """初始化直接处理器"""
        try:
            # 直接处理器不需要复杂的初始化
            self.is_initialized = True
            logger.info("✅ 直接数据库处理器初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 直接处理器初始化失败: {e}")
            return False
    
    def process_with_context(self, context: QueryContext) -> RAGResult:
        """使用QueryContext处理直接数据库查询"""
        
        query = context.original_query
        intent = context.intent_info.intent if context.intent_info else "unknown"
        
        try:
            # 尝试连接数据库并查询真实数据
            from app.database.nebula_connection import nebula_conn
            
            # 检查连接
            if not nebula_conn.is_connected():
                logger.warning("数据库未连接，使用模拟数据")
                return self._create_mock_result(query, intent)
            
            # 提取实体（简单实现）
            entities = self._extract_entities(query)
            
            # 构建查询语句
            if entities['players']:
                player_name = entities['players'][0]
                # 查询球员信息
                nql = f"MATCH (v:player) WHERE v.player.name CONTAINS '{player_name}' RETURN v.player.name AS name, v.player.age AS age, v.player.height AS height LIMIT 5"
                
                result = nebula_conn.execute_query(nql)
                if result and result.is_succeeded():
                    records = result.data()
                    if records:
                        return self._create_database_result(query, records, player_name)
            
            # 如果没有找到特定球员，尝试通用查询
            if "age" in query.lower() or "old" in query.lower():
                nql = "MATCH (v:player) RETURN v.player.name AS name, v.player.age AS age LIMIT 10"
                result = nebula_conn.execute_query(nql)
                if result and result.is_succeeded():
                    records = result.data()
                    if records:
                        return self._create_general_result(query, records, "age information")
            
            # 没有找到数据，返回空结果
            return self._create_empty_result(query, "No relevant data found")
            
        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            return self._create_mock_result(query, intent)
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """简单的实体提取"""
        players = []
        query_lower = query.lower()
        
        # 常见球员名字
        known_players = ['yao ming', 'kobe', 'lebron', 'jordan', 'durant', 'curry', 'harden']
        for player in known_players:
            if player in query_lower:
                # 标准化名字
                if player == 'yao ming':
                    players.append('Yao Ming')
                elif player == 'kobe':
                    players.append('Kobe Bryant')
                elif player == 'lebron':
                    players.append('LeBron James')
                elif player == 'jordan':
                    players.append('Michael Jordan')
                elif player == 'durant':
                    players.append('Kevin Durant')
                elif player == 'curry':
                    players.append('Stephen Curry')
                elif player == 'harden':
                    players.append('James Harden')
        
        return {'players': players, 'teams': []}
    
    def _create_database_result(self, query: str, records: list, player_name: str) -> RAGResult:
        """创建数据库查询结果"""
        retrieved_nodes = []
        context_parts = [f"Found information about {player_name}:"]
        
        for record in records:
            name = record.get('name', 'Unknown')
            age = record.get('age', 'Unknown')
            height = record.get('height', 'Unknown')
            
            node_data = {
                "name": name,
                "age": age,
                "height": height,
                "type": "player"
            }
            retrieved_nodes.append(node_data)
            
            info_parts = [f"- {name}"]
            if age and age != 'Unknown':
                info_parts.append(f", age: {age} years old")
            if height and height != 'Unknown':
                info_parts.append(f", height: {height}")
            
            context_parts.append("".join(info_parts))
        
        context_text = "\n".join(context_parts)
        
        return RAGResult(
            success=True,
            processor_used=self.processor_name,
            processing_strategy="database_query",
            context_text=context_text,
            retrieved_nodes=retrieved_nodes,
            confidence=0.9,
            metadata={
                "query_type": "database_lookup",
                "records_found": len(records),
                "player_searched": player_name
            }
        )
    
    def _create_general_result(self, query: str, records: list, info_type: str) -> RAGResult:
        """创建通用查询结果"""
        retrieved_nodes = []
        context_parts = [f"Found {info_type}:"]
        
        for record in records[:5]:  # 限制显示数量
            name = record.get('name', 'Unknown')
            age = record.get('age', 'Unknown')
            
            node_data = {
                "name": name,
                "age": age,
                "type": "player"
            }
            retrieved_nodes.append(node_data)
            
            if age and age != 'Unknown':
                context_parts.append(f"- {name}: {age} years old")
            else:
                context_parts.append(f"- {name}: age unknown")
        
        context_text = "\n".join(context_parts)
        
        return RAGResult(
            success=True,
            processor_used=self.processor_name,
            processing_strategy="general_database_query",
            context_text=context_text,
            retrieved_nodes=retrieved_nodes,
            confidence=0.8,
            metadata={
                "query_type": "general_lookup",
                "records_found": len(records)
            }
        )
    
    def _create_empty_result(self, query: str, reason: str) -> RAGResult:
        """创建空结果"""
        return RAGResult(
            success=False,
            processor_used=self.processor_name,
            processing_strategy="empty_result",
            context_text=f"Sorry, {reason}. The query content may not be within the knowledge base scope, or the database is temporarily unavailable.",
            retrieved_nodes=[],
            confidence=0.1,
            error_message=reason,
            metadata={
                "query_type": "failed_lookup",
                "reason": reason
            }
        )
    
    def _create_mock_result(self, query: str, intent: str) -> RAGResult:
        """创建模拟结果（当数据库不可用时）"""
        # 模拟数据库查询结果
        if "age" in query.lower() or "old" in query.lower():
            context_text = "Player age information (mock data):\nYao Ming: 38 years old\nJames Harden: 29 years old\nLeBron James: 37 years old"
            retrieved_nodes = [
                {"player": "Yao Ming", "age": 38, "position": "Center"},
                {"player": "James Harden", "age": 29, "position": "Guard"},
                {"player": "LeBron James", "age": 37, "position": "Forward"}
            ]
            confidence = 0.85  # 降低模拟数据的置信度
        elif "height" in query.lower() or "tall" in query.lower():
            context_text = "Player height information (mock data):\nYao Ming: 226cm\nShaquille O'Neal: 216cm\nKobe Bryant: 198cm"
            retrieved_nodes = [
                {"player": "Yao Ming", "height": "226cm", "position": "Center"},
                {"player": "Shaquille O'Neal", "height": "216cm", "position": "Center"},
                {"player": "Kobe Bryant", "height": "198cm", "position": "Guard"}
            ]
            confidence = 0.82
        else:
            context_text = f"Found general information related to query '{query}' (mock data)"
            retrieved_nodes = [{"info": "general_basketball_info", "relevance": 0.7}]
            confidence = 0.7
        
        return RAGResult(
            success=True,
            processor_used=self.processor_name,
            processing_strategy="mock_data_fallback",
            context_text=context_text,
            retrieved_nodes=retrieved_nodes,
            confidence=confidence,
            metadata={
                "query_type": "mock_data",
                "intent": intent,
                "node_count": len(retrieved_nodes),
                "data_source": "fallback"
            }
        )

    def _requires_intent(self) -> bool:
        """直接处理器建议有意图信息"""
        return True

class ChitchatContextProcessor(ContextAwareProcessor):
    """闲聊处理器 - QueryContext版本"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("chitchat_processor", config)
    
    def initialize(self) -> bool:
        """初始化闲聊处理器"""
        try:
            self.is_initialized = True
            logger.info("✅ 闲聊处理器初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 闲聊处理器初始化失败: {e}")
            return False
    
    def process_with_context(self, context: QueryContext) -> RAGResult:
        """使用QueryContext处理闲聊查询"""
        
        query = context.original_query
        
        # 生成闲聊响应的上下文
        context_text = f"这是一个关于篮球的闲聊查询：{query}"
        
        return RAGResult(
            success=True,
            processor_used=self.processor_name,
            processing_strategy="domain_chitchat",
            context_text=context_text,
            retrieved_nodes=[],
            confidence=0.8,
            metadata={
                "query_type": "chitchat",
                "domain": "basketball"
            }
        )
