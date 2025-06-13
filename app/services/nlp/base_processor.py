"""
NLP处理器基类
定义所有NLP组件的统一接口
"""
from abc import ABC, abstractmethod
from app.core.schemas import QueryContext
import logging

logger = logging.getLogger(__name__)

class BaseNLPProcessor(ABC):
    """NLP处理器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化处理器（加载模型、配置等）"""
        pass
    
    @abstractmethod 
    def process(self, context: QueryContext) -> QueryContext:
        """
        处理QueryContext，填充相应字段
        
        Args:
            context: 查询上下文对象
            
        Returns:
            QueryContext: 处理后的上下文对象（通常是同一个对象）
        """
        pass
    
    def _add_trace(self, context: QueryContext, action: str, details: dict = None):
        """添加处理追踪记录"""
        context.add_trace(
            component=f"nlp.{self.name}",
            action=action,
            data=details or {}
        )
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.initialized})"
