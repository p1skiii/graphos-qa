"""
NLP处理管道
将所有NLP组件串联起来，实现完整的文本处理流程
"""
from typing import List, Optional, Dict, Any
from app.core.schemas import QueryContext, QueryContextFactory
from .base_processor import BaseNLPProcessor
from .language_detector import LanguageDetector
from .tokenizer import Tokenizer
from .entity_extractor import EntityExtractor
from .intent_classifier import IntentClassifier
import logging

logger = logging.getLogger(__name__)

class NLPPipeline:
    """NLP处理管道 - 协调所有NLP组件的处理流程"""
    
    def __init__(self, 
                 language_detector: Optional[LanguageDetector] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 entity_extractor: Optional[EntityExtractor] = None,
                 intent_classifier: Optional[IntentClassifier] = None):
        """
        初始化NLP管道
        
        Args:
            language_detector: 语言检测器
            tokenizer: 分词器
            entity_extractor: 实体提取器
            intent_classifier: 意图分类器
        """
        self.processors: List[BaseNLPProcessor] = []
        
        # 使用提供的组件或创建默认组件
        self.language_detector = language_detector or LanguageDetector()
        self.tokenizer = tokenizer or Tokenizer()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.intent_classifier = intent_classifier or IntentClassifier()
        
        # 按顺序添加处理器
        self.processors = [
            self.language_detector,
            self.tokenizer,
            self.entity_extractor,
            self.intent_classifier
        ]
        
        self.initialized = False
        
    def initialize(self) -> bool:
        """初始化所有处理器"""
        try:
            failed_processors = []
            
            for processor in self.processors:
                if not processor.initialize():
                    failed_processors.append(processor.name)
            
            if failed_processors:
                logger.error(f"❌ NLP管道初始化失败，失败的组件: {failed_processors}")
                return False
            
            self.initialized = True
            logger.info(f"✅ NLP管道初始化成功 (组件数: {len(self.processors)})")
            return True
            
        except Exception as e:
            logger.error(f"❌ NLP管道初始化异常: {e}")
            return False
    
    def process(self, context: QueryContext) -> QueryContext:
        """
        完整的NLP处理流程
        
        Args:
            context: 查询上下文
            
        Returns:
            QueryContext: 处理完成的上下文
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("NLP管道未初始化")
        
        logger.debug(f"🔄 开始NLP处理: {context.original_query[:50]}...")
        
        try:
            # 逐个处理器处理
            for processor in self.processors:
                context = processor.process(context)
                
            logger.info(f"✅ NLP处理完成: 语言={context.language_info.original_language if context.language_info else 'unknown'}, "
                       f"意图={context.intent_info.intent if context.intent_info else 'unknown'}")
            
        except Exception as e:
            logger.error(f"❌ NLP处理失败: {e}")
            # 添加错误追踪
            context.add_trace("nlp_pipeline", "processing_error", {"error": str(e)})
        
        return context
    
    def process_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> QueryContext:
        """
        便捷方法：从字符串查询开始处理
        
        Args:
            query: 用户查询字符串
            user_context: 用户上下文（可选）
            
        Returns:
            QueryContext: 处理完成的上下文
        """
        # 创建初始上下文
        context = QueryContextFactory.create(query, user_context)
        
        # 处理
        return self.process(context)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        stats = {
            "initialized": self.initialized,
            "total_processors": len(self.processors),
            "processor_status": {}
        }
        
        for processor in self.processors:
            stats["processor_status"][processor.name] = {
                "initialized": processor.initialized,
                "class": processor.__class__.__name__
            }
        
        return stats
    
    def test_individual_components(self, test_query: str = "How old is Kobe Bryant?") -> Dict[str, Any]:
        """测试各个组件的独立功能"""
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("NLP管道未初始化")
        
        results = {}
        
        # 测试语言检测
        try:
            lang_result = self.language_detector.detect_language_only(test_query)
            results["language_detection"] = lang_result
        except Exception as e:
            results["language_detection"] = {"error": str(e)}
        
        # 测试分词
        try:
            tokens = self.tokenizer.tokenize_text(test_query)
            results["tokenization"] = {
                "token_count": len(tokens),
                "tokens": [{"text": t.text, "pos": t.pos, "lemma": t.lemma} for t in tokens[:10]]  # 只显示前10个
            }
        except Exception as e:
            results["tokenization"] = {"error": str(e)}
        
        # 测试实体提取
        try:
            entities = self.entity_extractor.extract_entities_from_text(test_query)
            results["entity_extraction"] = entities
        except Exception as e:
            results["entity_extraction"] = {"error": str(e)}
        
        # 测试意图分类
        try:
            intent = self.intent_classifier.classify_text_only(test_query)
            results["intent_classification"] = intent
        except Exception as e:
            results["intent_classification"] = {"error": str(e)}
        
        return results


def create_default_nlp_pipeline() -> NLPPipeline:
    """创建默认的NLP管道"""
    return NLPPipeline()


def create_english_only_pipeline() -> NLPPipeline:
    """创建仅处理英文的NLP管道（跳过语言检测）"""
    # 可以在这里自定义组件配置
    return NLPPipeline()
