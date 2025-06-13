"""
统一数据合约

定义了整个系统的标准化数据结构，作为“单一事实来源”
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

# =============================================================================
# Base Data Structures
# =============================================================================

@dataclass
class LanguageInfo:
    """语言检测和处理信息"""
    original_language: str              # 原始语言代码（zh/en）
    detected_confidence: float          # 语言检测置信度
    normalized_language: str = "en"     # 标准化语言（强制英语处理）
    translation_needed: bool = False    # 是否需要翻译
    
    def __post_init__(self):
        """自动设置翻译标志"""
        self.translation_needed = (self.original_language != "en")

@dataclass 
class EntityInfo:
    """实体提取信息"""
    players: List[str] = field(default_factory=list)        # 玩家名单
    teams: List[str] = field(default_factory=list)          # 队伍名称列表
    target_entities: List[str] = field(default_factory=list) # 所有目标实体（球员+队伍）
    
    # 保留用于其他模块的字段
    attributes: List[str] = field(default_factory=list)     # 属性类型列表（由IntentClassifier填充）
    numbers: List[str] = field(default_factory=list)        # 数字列表
    question_words: List[str] = field(default_factory=list) # 疑问词列表
    
    # 高级实体信息
    entity_relationships: Dict[str, Any] = field(default_factory=dict)  # 实体关系
    confidence_scores: Dict[str, float] = field(default_factory=dict)   # 实体置信度得分
    
    # 向后兼容属性
    @property
    def target_entity(self) -> Optional[str]:
        """向后兼容：返回第一个目标实体"""
        return self.target_entities[0] if self.target_entities else None

@dataclass
class IntentInfo:
    """意图分类信息"""
    intent: str                                             # 主要意图
    confidence: float                                       # 置信度
    all_scores: Dict[str, float] = field(default_factory=dict)  # 所有意图分数
    
    # 意图子分类
    query_type: Optional[str] = None                        # 查询类型
    attribute_type: Optional[str] = None                    # 属性类型
    complexity: str = "simple"                              # 查询复杂度
    direct_answer_expected: bool = True                     # 是否预期直接答案

@dataclass
class RAGResult:
    """RAG处理结果"""
    success: bool
    processor_used: str                                     # 处理器使用
    processing_strategy: str                                # 处理策略
    processing_time: float = 0.0                           # 处理时间
    
    # 核心结果数据
    context_text: str = ""                                  # 上下文文本
    retrieved_nodes: List[Dict[str, Any]] = field(default_factory=list)  # 已检索节点列表
    retrieved_nodes_count: int = 0                          # 已检索节点计数 (向后兼容)
    confidence: float = 0.0                                 # 结果置信度
    
    # 详细结果信息
    subgraph_summary: Dict[str, Any] = field(default_factory=dict)     # 子图摘要
    raw_data: Dict[str, Any] = field(default_factory=dict)             # 原始数据
    metadata: Dict[str, Any] = field(default_factory=dict)             # 元数据
    
    # 错误处理
    error_message: Optional[str] = None                     # 错误信息（正确名称）
    warnings: List[str] = field(default_factory=list)      # 警告信息
    
    # 向后兼容属性
    @property
    def contextualized_text(self) -> str:
        """向后兼容性：返回上下文文本"""
        return self.context_text
    
    @property 
    def error(self) -> Optional[str]:
        """向后兼容性：返回错误信息"""
        return self.error_message

@dataclass
class LLMResult:
    """LLM生成结果"""
    success: bool
    content: str = ""                                       #生成内容
    processing_time: float = 0.0                           # 处理时间
    
    # LLM详细信息
    model_used: Optional[str] = None                        # 使用的模型
    tokens_used: int = 0                                    # 使用的token
    generation_params: Dict[str, Any] = field(default_factory=dict)    # 生成参数
    
    # 质量评估
    quality_score: Optional[float] = None                   # 质量得分
    coherence_score: Optional[float] = None                 # 一致性得分
    
    # 错误处理
    error: Optional[str] = None                             # 错误信息
    fallback_used: bool = False                             # 是否使用了回退机制
    
    # 向后兼容属性
    @property
    def model_name(self) -> Optional[str]:
        """向后兼容性：返回model_used"""
        return self.model_used

# =============================================================================
# Core Context Object - System "Data Backbone"
# =============================================================================

@dataclass
class QueryContext:
    """
    全局请求上下文 - 穿越整个系统的“公文包”
    每个处理阶段在此对象中记录输入和输出，形成完整的请求追踪
    """
    
    # 基本信息
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 用户输入
    original_query: str = ""                                # User original query
    user_context: Dict[str, Any] = field(default_factory=dict)  # User context
    
    # 语言处理 - 由智能预处理器填充
    language_info: Optional[LanguageInfo] = None
    normalized_query: str = ""                              # Normalized query (English)
    
    # NLP处理结果 - 由NLP模块填充
    tokens: List[Any] = field(default_factory=list)        # 分词结果
    
    # 意图和实体 - 由统一意图分类器填充
    intent_info: Optional[IntentInfo] = None
    entity_info: Optional[EntityInfo] = None
    
    # 路由信息 - 由路由器填写
    routing_path: str = ""                                  # Routing path
    processor_selected: str = ""                            # Selected processor
    routing_reason: str = ""                                # Routing reason
    routing_time: float = 0.0                               # Routing time
    
    # RAG处理 - 由RAG处理器填充
    rag_result: Optional[RAGResult] = None
    
    # LLM生成 - 由LLM引擎填充
    llm_result: Optional[LLMResult] = None
    
    # 最终结果
    final_answer: str = ""                                  # 最终回答
    answer_language: str = "en"                             # 回答语言
    
    # 性能和质量指标
    total_processing_time: float = 0.0                      # 总处理时间
    cache_hit: bool = False                                 # 是否命中缓存
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # 质量指标
    
    # 错误和状态
    status: str = "processing"                              # 处理状态: processing/success/error
    errors: List[str] = field(default_factory=list)        # 错误列表
    warnings: List[str] = field(default_factory=list)      # 警告列表
    
    # 调试和追踪信息
    debug_info: Dict[str, Any] = field(default_factory=dict)  # 调试信息
    processing_trace: List[Dict[str, Any]] = field(default_factory=list)  # 处理轨迹
    
    # 向后兼容的属性
    @property
    def trace_log(self) -> List[Dict[str, Any]]:
        """向后兼容：返回processing_trace"""
        return self.processing_trace
    
    def add_trace(self, component: str, action: str, data: Dict[str, Any] = None):
        """添加处理轨迹记录"""
        trace_entry = {
            "timestamp": datetime.now(),
            "component": component,
            "action": action,
            "data": data or {}
        }
        self.processing_trace.append(trace_entry)
    
    def add_error(self, component: str, error: str):
        """添加错误信息"""
        error_msg = f"[{component}] {error}"
        self.errors.append(error_msg)
        self.status = "error"
        self.add_trace(component, "error", {"error": error})
    
    def add_warning(self, component: str, warning: str):
        """添加警告信息"""
        warning_msg = f"[{component}] {warning}"
        self.warnings.append(warning_msg)
        self.add_trace(component, "warning", {"warning": warning})
    
    def mark_success(self):
        """标记处理成功"""
        self.status = "success"
        self.add_trace("system", "completed", {"final_status": "success"})
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        return {
            "request_id": self.request_id,
            "total_time": self.total_processing_time,
            "status": self.status,
            "components_used": [trace["component"] for trace in self.processing_trace],
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "cache_hit": self.cache_hit
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于API响应"""
        return {
            "request_id": self.request_id,
            "query": self.original_query,
            "final_answer": self.final_answer,
            "intent": self.intent_info.intent if self.intent_info else "unknown",
            "processor_used": self.processor_selected,
            "processing_time": self.total_processing_time,
            "status": self.status,
            "language": {
                "original": self.language_info.original_language if self.language_info else "unknown",
                "answer": self.answer_language
            },
            "metadata": {
                "routing_path": self.routing_path,
                "routing_reason": self.routing_reason,
                "cache_hit": self.cache_hit,
                "quality_metrics": self.quality_metrics,
                "errors": self.errors,
                "warnings": self.warnings
            }
        }

# =============================================================================
# 工厂类 - 统一创建和管理上下文对象
# =============================================================================

class QueryContextFactory:
    """QueryContext工厂类，用于创建和管理上下文对象"""
    
    @staticmethod
    def create(query: str, user_context: Dict[str, Any] = None) -> QueryContext:
        """创建新的查询上下文"""
        context = QueryContext(
            original_query=query,
            user_context=user_context or {}
        )
        
        # 添加初始追踪记录
        context.add_trace("factory", "created", {
            "query_length": len(query),
            "has_user_context": bool(user_context)
        })
        
        return context
    
    @staticmethod
    def create_from_dict(data: Dict[str, Any]) -> QueryContext:
        """从字典创建上下文（用于反序列化）"""
        # 这里可以添加更复杂的反序列化逻辑
        context = QueryContext(
            original_query=data.get("query", ""),
            user_context=data.get("user_context", {})
        )
        
        # 恢复其他字段...
        if "language_info" in data:
            context.language_info = LanguageInfo(**data["language_info"])
        
        return context
    
    @staticmethod
    def validate_context(context: QueryContext) -> List[str]:
        """验证上下文对象的完整性"""
        validation_errors = []
        
        # 基础验证
        if not context.original_query.strip():
            validation_errors.append("Original query cannot be empty")
        
        # Language information validation
        if context.language_info and context.language_info.detected_confidence < 0.5:
            validation_errors.append("Language detection confidence too low")
        
        # Intent information validation
        if context.intent_info and context.intent_info.confidence < 0.3:
            validation_errors.append("Intent classification confidence too low")
        
        # RAG result validation
        if context.rag_result and not context.rag_result.success and not context.rag_result.error:
            validation_errors.append("RAG processing failed but no error message provided")
        
        return validation_errors
    
    @staticmethod
    def create_mock_context(query: str = "How old is Yao Ming?") -> QueryContext:
        """创建模拟上下文对象（用于测试）"""
        context = QueryContextFactory.create(query)
        
        # 填充模拟数据
        context.language_info = LanguageInfo(
            original_language="en",
            detected_confidence=0.95
        )
        
        context.entity_info = EntityInfo(
            players=["Yao Ming"],
            attributes=["age"],
            target_entity="Yao Ming"
        )
        
        context.intent_info = IntentInfo(
            intent="ATTRIBUTE_QUERY",
            confidence=0.92,
            attribute_type="age"
        )
        
        return context

# =============================================================================
# 数据验证和转换工具
# =============================================================================

class DataValidator:
    """数据验证工具类"""
    
    @staticmethod
    def validate_language_info(lang_info: LanguageInfo) -> bool:
        """验证语言信息"""
        return (
            lang_info.original_language in ["zh", "en", "auto"] and
            0.0 <= lang_info.detected_confidence <= 1.0
        )
    
    @staticmethod
    def validate_intent_info(intent_info: IntentInfo) -> bool:
        """验证意图信息"""
        valid_intents = [
            "ATTRIBUTE_QUERY",
            "SIMPLE_RELATION_QUERY", 
            "COMPLEX_RELATION_QUERY",
            "COMPARATIVE_QUERY",
            "DOMAIN_CHITCHAT",
            "OUT_OF_DOMAIN"
        ]
        
        return (
            intent_info.intent in valid_intents and
            0.0 <= intent_info.confidence <= 1.0
        )
    
    @staticmethod
    def validate_entity_info(entity_info: EntityInfo) -> bool:
        """验证实体信息"""
        if not isinstance(entity_info, EntityInfo):
            return False
        
        # 检查至少有一些实体信息
        has_entities = (
            entity_info.players or 
            entity_info.teams or 
            entity_info.attributes or
            entity_info.question_words
        )
        
        return has_entities
    
    @staticmethod
    def validate_rag_result(rag_result: RAGResult) -> bool:
        """验证RAG结果"""
        if not isinstance(rag_result, RAGResult):
            return False
        
        # 基础字段检查
        if not hasattr(rag_result, 'success') or not hasattr(rag_result, 'processor_used'):
            return False
        
        # 如果成功，检查结果内容
        if rag_result.success:
            return bool(rag_result.context_text or rag_result.retrieved_nodes)
        
        return True
    
    @staticmethod
    def validate_llm_result(llm_result: LLMResult) -> bool:
        """验证LLM结果"""
        if not isinstance(llm_result, LLMResult):
            return False
        
        # 基础字段检查
        if not hasattr(llm_result, 'success'):
            return False
        
        # 如果成功，检查内容
        if llm_result.success:
            return bool(llm_result.content)
        
        return True

class DataConverter:
    """数据转换工具类"""
    
    @staticmethod
    def context_to_legacy_format(context: QueryContext) -> Dict[str, Any]:
        """将新格式转换为旧系统兼容格式"""
        return {
            "query": context.original_query,
            "intent": context.intent_info.intent if context.intent_info else "unknown",
            "entities": {
                "players": context.entity_info.players if context.entity_info else [],
                "teams": context.entity_info.teams if context.entity_info else [],
                "attributes": context.entity_info.attributes if context.entity_info else []
            },
            "rag_result": context.rag_result.__dict__ if context.rag_result else {},
            "llm_response": context.llm_result.content if context.llm_result else None
        }
    
    @staticmethod
    def legacy_to_context_format(legacy_data: Dict[str, Any]) -> QueryContext:
        """将旧格式转换为新的上下文格式"""
        context = QueryContextFactory.create(legacy_data.get("query", ""))
        
        # 转换意图信息
        if "intent" in legacy_data:
            context.intent_info = IntentInfo(
                intent=legacy_data["intent"],
                confidence=0.8  # 默认置信度
            )
        
        # 转换实体信息
        if "entities" in legacy_data:
            entities = legacy_data["entities"]
            context.entity_info = EntityInfo(
                players=entities.get("players", []),
                teams=entities.get("teams", []),
                attributes=entities.get("attributes", [])
            )
        
        return context
