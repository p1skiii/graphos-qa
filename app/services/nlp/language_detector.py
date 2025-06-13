"""
语言检测器
检测查询语言并填充language_info字段
"""
import langdetect
from typing import Dict, Any
from app.core.schemas import QueryContext, LanguageInfo
from .base_processor import BaseNLPProcessor
import logging

logger = logging.getLogger(__name__)

class LanguageDetector(BaseNLPProcessor):
    """语言检测器 - 检测输入文本的语言"""
    
    def __init__(self):
        super().__init__("language_detector")
        self.confidence_threshold = 0.7
        
    def initialize(self) -> bool:
        """初始化语言检测器"""
        try:
            # langdetect库不需要特殊初始化
            self.initialized = True
            logger.info(f"✅ {self.name} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ {self.name} 初始化失败: {e}")
            return False
    
    def process(self, context: QueryContext) -> QueryContext:
        """
        检测语言并填充language_info
        
        Args:
            context: 查询上下文
            
        Returns:
            QueryContext: 填充了language_info的上下文
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError(f"{self.name} 未初始化")
        
        self._add_trace(context, "start_detection")
        
        try:
            # 检测语言
            query = context.original_query.strip()
            if not query:
                raise ValueError("查询内容不能为空，请输入有效问题")
            
            if len(query) < 2:
                raise ValueError("查询内容太短，请输入完整问题") 
                
            try:
                detected_lang = langdetect.detect(query)
            except langdetect.LangDetectException:
                raise ValueError("无法检测语言，可能是文本过短或不明确")
            
            # 严格检查：只允许英文
            if detected_lang != "en":
                raise ValueError(f"当前只支持英文查询，检测到语言：{detected_lang}")
            
            # 创建LanguageInfo对象（固定英文）
            language_info = LanguageInfo(
                original_language="en",
                detected_confidence=1.0,
                normalized_language="en",        # 强制英文
                translation_needed=False         # 不需要翻译
            )
            
            context.language_info = language_info
            self._add_trace(context, "english_confirmed")
            
        except Exception as e:
            logger.error(f"❌ 语言检测失败: {e}")
            # 直接抛出异常，不要默认处理
            raise e

        return context