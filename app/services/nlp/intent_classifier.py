"""
意图分类器
基于语法决策树的篮球查询意图分类 - 重构版本
"""
from typing import Dict, List, Tuple, Optional
from app.core.schemas import QueryContext, IntentInfo
from .base_processor import BaseNLPProcessor
from .tokenizer import Token
import logging

logger = logging.getLogger(__name__)

class IntentClassifier(BaseNLPProcessor):
    """意图分类器 - 基于语法决策树的篮球查询意图分类"""
    
    def __init__(self):
        super().__init__("intent_classifier")
        self.nlp = None
        
        # 语法决策树规则 - 基于spaCy语法分析
        self.attribute_keywords = {
            'age': {'how old', 'age', 'years old', 'born', 'birth'},
            'height': {'how tall', 'height', 'tall', 'cm', 'feet', 'inches', 'ft'},
            'weight': {'how heavy', 'weight', 'heavy', 'kg', 'pounds', 'lbs'},
            'position': {'position', 'guard', 'forward', 'center', 'play'},
            'team': {'team', 'club', 'play for'},
            'stats': {'points', 'rebounds', 'assists', 'statistics', 'stats', 'average'}
        }
        
        self.comparison_indicators = {
            'compare', 'better', 'vs', 'versus', 'who is', 'taller', 'older', 'stronger'
        }
        
        self.basketball_domain = {
            'basketball', 'nba', 'game', 'player', 'team', 'coach', 'season', 'playoff'
        }
        
    def initialize(self) -> bool:
        """初始化意图分类器"""
        try:
            # 加载spaCy模型
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.initialized = True
            logger.info(f"✅ {self.name} 初始化成功 (语法决策树模式)")
            return True
        except Exception as e:
            logger.error(f"❌ {self.name} 初始化失败: {e}")
            return False
    
    def process(self, context: QueryContext) -> QueryContext:
        """
        语法决策树意图分类
        
        Args:
            context: 查询上下文
            
        Returns:
            QueryContext: 填充了intent_info的上下文
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError(f"{self.name} 未初始化")
        
        self._add_trace(context, "start_grammar_classification")
        
        try:
            # 使用spaCy进行语法分析
            text = context.original_query
            doc = self.nlp(text)
            entity_info = context.entity_info
            
            # 语法决策树分类
            intent, confidence, attribute_type = self._grammar_decision_tree(doc, entity_info)
            
            # 确定复杂度
            complexity = self._analyze_complexity(doc, entity_info)
            
            # 创建IntentInfo对象
            intent_info = IntentInfo(
                intent=intent,
                confidence=confidence,
                query_type=self._get_query_subtype(intent),
                attribute_type=attribute_type,
                complexity=complexity,
                direct_answer_expected=intent in ['ATTRIBUTE_QUERY', 'SIMPLE_RELATION_QUERY']
            )
            
            context.intent_info = intent_info
            self._add_trace(context, "grammar_classification_complete", {
                "intent": intent,
                "confidence": confidence,
                "attribute_type": attribute_type,
                "complexity": complexity
            })
            
            logger.debug(f"🎯 语法决策树分类: {intent} (confidence: {confidence:.2f}, attr: {attribute_type})")
            
        except Exception as e:
            logger.error(f"❌ 语法分类失败: {e}")
            context.intent_info = IntentInfo(
                intent="OUT_OF_DOMAIN",
                confidence=0.5,
                query_type="unknown",
                attribute_type="unknown",
                complexity="unknown",
                direct_answer_expected=False
            )
            self._add_trace(context, "grammar_classification_error", {"error": str(e)})
        
        return context
    
    def _grammar_decision_tree(self, doc, entity_info) -> Tuple[str, float, str]:
        """
        核心语法决策树：基于spaCy语法分析的智能分类
        
        决策流程：
        1. 领域检查 (篮球相关 vs 领域外)
        2. 句型分析 (疑问句 vs 陈述句)
        3. 实体数量分析 (单实体 vs 多实体)
        4. 语法模式识别 (WH-词, 比较词, 属性词)
        """
        text_lower = doc.text.lower()
        
        # 第1层: 领域检查
        is_basketball_domain = self._is_basketball_domain(doc, entity_info)
        if not is_basketball_domain:
            return "OUT_OF_DOMAIN", 0.8, "unknown"
        
        # 第2层: 句型分析
        is_question = self._is_question(doc)
        has_entities = entity_info and (entity_info.players or entity_info.teams)
        
        # 第3层: 实体数量分析
        total_entities = 0
        if entity_info:
            total_entities = len(entity_info.players) + len(entity_info.teams)
        
        # 第4层: 语法模式决策
        if not has_entities:
            # 无实体 -> 闲聊或领域外
            if any(word in text_lower for word in ['favorite', 'best', 'greatest', 'think', 'love']):
                return "DOMAIN_CHITCHAT", 0.9, "unknown"
            else:
                return "OUT_OF_DOMAIN", 0.7, "unknown"
        
        # 有实体的情况
        if total_entities >= 3 or any(word in text_lower for word in self.comparison_indicators):
            # 多实体或比较词 -> 比较查询
            attribute_type = self._detect_attribute_type(doc)
            return "COMPARATIVE_QUERY", 0.9, attribute_type
        
        elif total_entities == 2:
            # 双实体 -> 关系查询
            if any(word in text_lower for word in ['with', 'together', 'teammate', 'coach']):
                return "SIMPLE_RELATION_QUERY", 0.85, "unknown"
            else:
                # 也可能是比较查询
                return "COMPARATIVE_QUERY", 0.8, self._detect_attribute_type(doc)
        
        else:  # total_entities == 1
            # 单实体处理
            attribute_type = self._detect_attribute_type(doc)
            
            # 检查是否是关系查询 (单实体但询问关系)
            relation_indicators = ['coach', 'coached', 'teammate', 'manager', 'mentor']
            if any(word in text_lower for word in relation_indicators):
                return "SIMPLE_RELATION_QUERY", 0.8, "unknown"
            
            # 单实体属性查询
            if attribute_type != "unknown":
                return "ATTRIBUTE_QUERY", 0.9, attribute_type
            else:
                # 可能是简单的闲聊
                return "DOMAIN_CHITCHAT", 0.7, "unknown"
    
    def _is_basketball_domain(self, doc, entity_info) -> bool:
        """检查是否是篮球领域"""
        text_lower = doc.text.lower()
        
        # 有篮球实体 -> 肯定是篮球领域
        if entity_info and (entity_info.players or entity_info.teams):
            return True
        
        # 包含篮球关键词
        if any(word in text_lower for word in self.basketball_domain):
            return True
        
        return False
    
    def _is_question(self, doc) -> bool:
        """检查是否是疑问句"""
        text = doc.text.strip()
        
        # 以问号结尾
        if text.endswith('?'):
            return True
        
        # 以疑问词开头
        wh_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose'}
        first_token = doc[0].text.lower()
        if first_token in wh_words:
            return True
        
        # 包含助动词提问模式 (do, does, did, is, are, was, were)
        aux_verbs = {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would'}
        if first_token in aux_verbs:
            return True
        
        return False
    
    def _detect_attribute_type(self, doc) -> str:
        """基于语法模式检测属性类型"""
        text_lower = doc.text.lower()
        
        for attr_type, keywords in self.attribute_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return attr_type
        
        return "unknown"
    
    def _analyze_complexity(self, doc, entity_info) -> str:
        """分析查询复杂度"""
        text_lower = doc.text.lower()
        
        # 复杂指标
        complex_indicators = ['analysis', 'statistics', 'career comparison', 'detailed']
        if any(indicator in text_lower for indicator in complex_indicators):
            return "complex"
        
        # 中等复杂度指标
        medium_indicators = ['compare', 'versus', 'better', 'vs']
        if any(indicator in text_lower for indicator in medium_indicators):
            return "medium"
        
        # 基于实体数量
        if entity_info:
            total_entities = len(entity_info.players) + len(entity_info.teams)
            if total_entities >= 3:
                return "complex"
            elif total_entities == 2:
                return "medium"
        
        return "simple"
    
    def _get_query_subtype(self, intent: str) -> str:
        """获取查询子类型"""
        type_mapping = {
            'ATTRIBUTE_QUERY': 'single_entity_attribute',
            'SIMPLE_RELATION_QUERY': 'dual_entity_relation',
            'COMPARATIVE_QUERY': 'multi_entity_comparison',
            'DOMAIN_CHITCHAT': 'basketball_general',
            'OUT_OF_DOMAIN': 'non_basketball'
        }
        return type_mapping.get(intent, 'unknown')
    
    def classify_text_only(self, text: str) -> Dict:
        """独立分类功能（用于测试）"""
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError(f"{self.name} 未初始化")
        
        doc = self.nlp(text)
        intent, confidence, attribute_type = self._grammar_decision_tree(doc, None)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "attribute_type": attribute_type,
            "method": "grammar_decision_tree"
        }
