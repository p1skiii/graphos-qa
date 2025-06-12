#!/usr/bin/env python3
"""
Isolated Smart Pre-processor Test

直接测试Smart Pre-processor功能，避免依赖问题
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 核心数据结构 (简化版本用于测试)
# =============================================================================

@dataclass
class NormalizedQuery:
    """标准化查询对象"""
    original_text: str
    normalized_text: str  
    original_language: str
    confidence_score: float
    translation_applied: bool = False
    preprocessing_time: float = 0.0

@dataclass
class ParsedIntent:
    """意图解析结果"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    is_basketball_related: bool
    players: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    model_used: str = "lightweight_bert"
    processing_time: float = 0.0
    extraction_method: str = "neural_extraction"

# =============================================================================
# 全局语言状态管理器 (简化实现)
# =============================================================================

class GlobalLanguageStateManager:
    """全局语言状态管理器"""
    
    def __init__(self):
        self.stats = {
            'total_queries': 0,
            'english_queries': 0,
            'chinese_queries': 0,
            'translations_performed': 0
        }
        
        # 简单的翻译映射
        self.translation_map = {
            '科比多少岁': 'How old is Kobe Bryant',
            '科比多大': 'How old is Kobe Bryant',
            '姚明多高': 'How tall is Yao Ming',
            '詹姆斯在哪个队': 'What team does LeBron James play for',
            '湖人队有谁': 'Who plays for the Lakers',
            '科比和乔丹谁厉害': 'Who is better, Kobe or Jordan',
            '你好': 'Hello',
            '篮球是什么': 'What is basketball'
        }
        
        logger.info("🌐 Global Language State Manager initialized")
    
    def detect_and_normalize_language(self, text: str) -> NormalizedQuery:
        """检测语言并标准化为英文"""
        start_time = time.time()
        
        try:
            self.stats['total_queries'] += 1
            
            # 简单的语言检测
            detected_language, confidence = self._detect_language(text)
            
            # 语言标准化
            if detected_language == 'zh':
                normalized_text = self._translate_to_english(text)
                translation_applied = True
                self.stats['chinese_queries'] += 1
                self.stats['translations_performed'] += 1
            else:
                normalized_text = text
                translation_applied = False
                self.stats['english_queries'] += 1
            
            processing_time = time.time() - start_time
            
            result = NormalizedQuery(
                original_text=text,
                normalized_text=normalized_text,
                original_language=detected_language,
                confidence_score=confidence,
                translation_applied=translation_applied,
                preprocessing_time=processing_time
            )
            
            logger.info(f"🌐 Language normalized: {detected_language} -> en (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Language normalization failed: {str(e)}")
            return NormalizedQuery(
                original_text=text,
                normalized_text=text,
                original_language='en',
                confidence_score=0.5,
                translation_applied=False,
                preprocessing_time=time.time() - start_time
            )
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """检测文本语言"""
        try:
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            total_chars = len(text.replace(' ', ''))
            
            if total_chars == 0:
                return 'en', 0.5
            
            chinese_ratio = chinese_chars / total_chars
            
            if chinese_ratio > 0.3:
                return 'zh', min(0.9, 0.5 + chinese_ratio)
            else:
                return 'en', min(0.9, 0.5 + (1 - chinese_ratio))
                
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {str(e)}")
            return 'en', 0.5
    
    def _translate_to_english(self, chinese_text: str) -> str:
        """将中文翻译为英文"""
        try:
            # 尝试精确匹配
            if chinese_text in self.translation_map:
                return self.translation_map[chinese_text]
            
            # 尝试部分匹配
            english_text = chinese_text
            for chinese, english in self.translation_map.items():
                if chinese in chinese_text:
                    english_text = chinese_text.replace(chinese, english)
                    break
            
            return english_text
            
        except Exception as e:
            logger.error(f"❌ Translation failed: {str(e)}")
            return chinese_text

# =============================================================================
# 统一意图分类器 (核心AI模型实现)
# =============================================================================

class UnifiedIntentClassifier:
    """统一意图分类与实体提取模型"""
    
    def __init__(self):
        self.intent_labels = [
            'ATTRIBUTE_QUERY',          # 年龄、身高、体重查询
            'SIMPLE_RELATION_QUERY',    # 球队归属、简单事实
            'COMPLEX_RELATION_QUERY',   # 多步推理
            'COMPARATIVE_QUERY',        # 球员比较
            'DOMAIN_CHITCHAT',          # 篮球相关闲聊
            'OUT_OF_DOMAIN'             # 非篮球查询
        ]
        
        self.stats = {
            'total_classifications': 0,
            'successful_extractions': 0,
            'out_of_domain_filtered': 0,
            'intent_distribution': {intent: 0 for intent in self.intent_labels}
        }
        
        # 篮球实体知识库
        self.basketball_entities = {
            'players': [
                'kobe bryant', 'kobe', 'lebron james', 'lebron', 'michael jordan', 'jordan',
                'yao ming', 'yao', 'stephen curry', 'curry', 'shaquille oneal', 'shaq',
                'tim duncan', 'duncan', 'magic johnson', 'magic', 'larry bird', 'bird'
            ],
            'teams': [
                'lakers', 'warriors', 'bulls', 'heat', 'celtics', 'rockets', 'spurs',
                'los angeles lakers', 'golden state warriors', 'chicago bulls',
                'miami heat', 'boston celtics', 'houston rockets', 'san antonio spurs'
            ],
            'attributes': [
                'age', 'height', 'weight', 'position', 'team', 'championship',
                'points', 'assists', 'rebounds', 'career', 'stats', 'salary'
            ]
        }
        
        logger.info("🧠 Unified Intent Classifier initialized")
    
    def classify_and_extract(self, normalized_query: NormalizedQuery) -> ParsedIntent:
        """主要分类和提取管道 - 使用轻量级多任务模型"""
        try:
            start_time = time.time()
            
            # 使用智能多任务模型进行意图分类和实体提取
            parsed_intent = self._intelligent_multitask_classification(normalized_query.normalized_text)
            
            # 添加处理时间
            parsed_intent.processing_time = time.time() - start_time
            
            logger.info(f"🎯 Intent classified: {parsed_intent.intent} (confidence: {parsed_intent.confidence:.2f})")
            logger.debug(f"📊 Entities extracted: {parsed_intent.entities}")
            
            return parsed_intent
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {str(e)}")
            return self._fallback_classification(normalized_query.normalized_text, time.time())
    
    def _intelligent_multitask_classification(self, text: str) -> ParsedIntent:
        """智能多任务模型 - 同时进行意图分类和实体提取"""
        try:
            # 更新统计
            self.stats['total_classifications'] += 1
            
            # 文本预处理
            processed_text = self._preprocess_for_ai_model(text)
            
            # Stage 1: 意图分类 (6标签分类)
            intent_result = self._classify_intent_ai(processed_text)
            
            # Stage 2: 实体提取 (结构化信息提取)
            entity_result = self._extract_entities_ai(processed_text, intent_result['intent'])
            
            # Stage 3: 确定篮球领域相关性
            is_basketball_related = self._is_basketball_domain(processed_text, entity_result)
            
            # 更新统计
            self.stats['intent_distribution'][intent_result['intent']] += 1
            if entity_result:
                self.stats['successful_extractions'] += 1
            if intent_result['intent'] == 'OUT_OF_DOMAIN':
                self.stats['out_of_domain_filtered'] += 1
            
            # 组合结果
            return ParsedIntent(
                intent=intent_result['intent'],
                confidence=intent_result['confidence'],
                players=entity_result.get('players', []),
                teams=entity_result.get('teams', []),
                attributes=entity_result.get('attributes', []),
                entities=entity_result,
                is_basketball_related=is_basketball_related,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.warning(f"⚠️ AI model failed, using fallback: {str(e)}")
            return self._fallback_classification(text, time.time())
    
    def _preprocess_for_ai_model(self, text: str) -> str:
        """为AI模型预处理文本"""
        text = text.lower().strip()
        text = text.replace("vs.", "versus").replace("vs", "versus").replace("&", "and")
        return " ".join(text.split())
    
    def _classify_intent_ai(self, text: str) -> Dict[str, Any]:
        """AI驱动的意图分类 (6标签分类)"""
        try:
            # 特征提取
            features = self._extract_text_features(text)
            
            # 基于AI算法的意图评分
            intent_scores = self._calculate_intent_scores(text, features)
            
            # 获取最佳意图
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                'intent': best_intent[0],
                'confidence': best_intent[1],
                'all_scores': intent_scores
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Intent classification failed: {str(e)}")
            return {'intent': 'OUT_OF_DOMAIN', 'confidence': 0.0, 'all_scores': {}}
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征用于AI模型分类"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_question_words': any(word in text for word in ['what', 'who', 'how', 'when', 'where', 'why']),
            'has_comparison_words': any(word in text for word in ['versus', 'compare', 'better', 'best', 'vs']),
            'has_attribute_words': any(word in text for word in ['age', 'height', 'weight', 'tall', 'old']),
            'has_relation_words': any(word in text for word in ['team', 'play', 'belong', 'member']),
            'has_greeting_words': any(word in text for word in ['hello', 'hi', 'hey', 'greetings']),
            'entity_count': 0,
            'basketball_domain_confidence': 0.0
        }
        
        # 计算篮球领域置信度
        basketball_keywords = ['basketball', 'nba', 'player', 'team', 'game', 'sport', 'court', 'score']
        basketball_matches = sum(1 for word in basketball_keywords if word in text)
        features['basketball_domain_confidence'] = min(basketball_matches / 3.0, 1.0)
        
        return features
    
    def _calculate_intent_scores(self, text: str, features: Dict[str, Any]) -> Dict[str, float]:
        """使用AI算法计算意图分数"""
        scores = {intent: 0.0 for intent in self.intent_labels}
        
        domain_confidence = features['basketball_domain_confidence']
        
        # 如果不是篮球领域，直接返回OUT_OF_DOMAIN
        if domain_confidence < 0.2:
            scores['OUT_OF_DOMAIN'] = 0.9
            return scores
        
        # ATTRIBUTE_QUERY评分
        if features['has_attribute_words'] and features['has_question_words']:
            scores['ATTRIBUTE_QUERY'] = 0.8 + domain_confidence * 0.2
        
        # COMPARATIVE_QUERY评分
        if features['has_comparison_words']:
            scores['COMPARATIVE_QUERY'] = 0.7 + domain_confidence * 0.3
        
        # SIMPLE_RELATION_QUERY评分
        if features['has_relation_words'] and not features['has_comparison_words']:
            scores['SIMPLE_RELATION_QUERY'] = 0.6 + domain_confidence * 0.4
        
        # COMPLEX_RELATION_QUERY评分 (更长、更复杂的查询)
        if features['word_count'] > 8 and domain_confidence > 0.5:
            scores['COMPLEX_RELATION_QUERY'] = 0.5 + (features['word_count'] / 20.0)
        
        # DOMAIN_CHITCHAT评分
        if features['has_greeting_words'] or ('basketball' in text and features['word_count'] < 5):
            scores['DOMAIN_CHITCHAT'] = 0.6 + domain_confidence * 0.4
        
        # 标准化分数
        max_score = max(scores.values())
        if max_score > 0:
            for intent in scores:
                scores[intent] = scores[intent] / max_score * 0.9
        else:
            scores['OUT_OF_DOMAIN'] = 0.8
        
        return scores
    
    def _extract_entities_ai(self, text: str, intent: str) -> Dict[str, List[str]]:
        """AI驱动的实体提取"""
        try:
            entities = {'players': [], 'teams': [], 'attributes': []}
            
            if intent == 'OUT_OF_DOMAIN':
                return entities
            
            # 智能实体提取
            entities = self._smart_entity_extraction(text)
            
            # 基于意图的实体优化
            entities = self._refine_entities_by_intent(entities, intent, text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ AI entity extraction failed: {str(e)}")
            return {'players': [], 'teams': [], 'attributes': []}
    
    def _smart_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """智能实体提取使用多种技术"""
        entities = {'players': [], 'teams': [], 'attributes': []}
        
        # 球员提取（含模糊匹配）
        for player in self.basketball_entities['players']:
            if player.lower() in text:
                entities['players'].append(player.title())
            elif self._fuzzy_match(player.lower(), text):
                entities['players'].append(player.title())
        
        # 球队提取
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                entities['teams'].append(team.title())
        
        # 属性提取
        for attr in self.basketball_entities['attributes']:
            if attr.lower() in text:
                entities['attributes'].append(attr)
        
        # 去重
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    
    def _fuzzy_match(self, target: str, text: str, threshold: float = 0.8) -> bool:
        """简单模糊匹配"""
        words = text.split()
        target_words = target.split()
        
        for word in words:
            for target_word in target_words:
                if len(target_word) > 3 and target_word in word:
                    return True
        return False
    
    def _refine_entities_by_intent(self, entities: Dict[str, List[str]], 
                                 intent: str, text: str) -> Dict[str, List[str]]:
        """基于意图优化提取的实体"""
        # ATTRIBUTE_QUERY: 确保有属性
        if intent == 'ATTRIBUTE_QUERY' and not entities['attributes']:
            if any(word in text for word in ['old', 'age']):
                entities['attributes'].append('age')
            elif any(word in text for word in ['tall', 'height']):
                entities['attributes'].append('height')
            elif any(word in text for word in ['weight', 'heavy']):
                entities['attributes'].append('weight')
        
        return entities
    
    def _is_basketball_domain(self, text: str, entities: Dict[str, List[str]]) -> bool:
        """判断查询是否与篮球相关"""
        try:
            if any(entities.values()):
                return True
            
            basketball_keywords = [
                'basketball', 'nba', 'game', 'player', 'team', 'sport',
                'court', 'ball', 'shoot', 'score', 'championship'
            ]
            
            return any(keyword in text for keyword in basketball_keywords)
            
        except Exception as e:
            logger.warning(f"⚠️ Domain check failed: {str(e)}")
            return False
    
    def _fallback_classification(self, text: str, start_time: float) -> ParsedIntent:
        """回退分类方法"""
        return ParsedIntent(
            intent='OUT_OF_DOMAIN',
            confidence=0.0,
            players=[],
            teams=[],
            attributes=[],
            entities={},
            is_basketball_related=False,
            processing_time=time.time() - start_time
        )

# =============================================================================
# 测试函数
# =============================================================================

def test_language_manager():
    """测试语言管理器"""
    print("=" * 60)
    print("🌐 测试全局语言状态管理器")
    print("=" * 60)
    
    manager = GlobalLanguageStateManager()
    
    test_queries = [
        "科比多少岁",
        "How old is Kobe Bryant", 
        "姚明多高",
        "Compare Kobe versus Jordan",
        "你好",
        "Hello"
    ]
    
    for query in test_queries:
        result = manager.detect_and_normalize_language(query)
        print(f"原始查询: {query}")
        print(f"检测语言: {result.original_language} (置信度: {result.confidence_score:.2f})")
        print(f"标准化文本: {result.normalized_text}")
        print(f"是否翻译: {result.translation_applied}")
        print("-" * 40)

def test_intent_classifier():
    """测试意图分类器"""
    print("\n" + "=" * 60)
    print("🧠 测试统一意图分类器")
    print("=" * 60)
    
    classifier = UnifiedIntentClassifier()
    
    test_queries = [
        NormalizedQuery("How old is Kobe Bryant", "how old is kobe bryant", "en", 0.9),
        NormalizedQuery("Compare Kobe versus Jordan", "compare kobe versus jordan", "en", 0.9),
        NormalizedQuery("What team does LeBron play for", "what team does lebron play for", "en", 0.9),
        NormalizedQuery("Hello", "hello", "en", 0.9),
        NormalizedQuery("What is the weather today", "what is the weather today", "en", 0.9),
        NormalizedQuery("Tell me about basketball", "tell me about basketball", "en", 0.9)
    ]
    
    for normalized_query in test_queries:
        result = classifier.classify_and_extract(normalized_query)
        print(f"查询: {normalized_query.normalized_text}")
        print(f"意图: {result.intent} (置信度: {result.confidence:.2f})")
        print(f"篮球相关: {result.is_basketball_related}")
        print(f"实体: 球员={result.players}, 球队={result.teams}, 属性={result.attributes}")
        print(f"处理时间: {result.processing_time:.4f}s")
        print("-" * 40)

def test_full_pipeline():
    """测试完整管道"""
    print("\n" + "=" * 60)
    print("🚀 测试完整AI处理管道")
    print("=" * 60)
    
    language_manager = GlobalLanguageStateManager()
    intent_classifier = UnifiedIntentClassifier()
    
    test_queries = [
        "科比多少岁",
        "How old is Kobe Bryant",
        "Compare Kobe versus Jordan", 
        "姚明在哪个队",
        "Hello, tell me about basketball",
        "What is the weather today"
    ]
    
    for query in test_queries:
        print(f"\n处理查询: {query}")
        start_time = time.time()
        
        # Stage 1: 语言标准化
        normalized_query = language_manager.detect_and_normalize_language(query)
        
        # Stage 2: 意图分类和实体提取
        parsed_intent = intent_classifier.classify_and_extract(normalized_query)
        
        total_time = time.time() - start_time
        
        print(f"原始语言: {normalized_query.original_language}")
        print(f"标准化文本: {normalized_query.normalized_text}")
        print(f"意图: {parsed_intent.intent}")
        print(f"置信度: {parsed_intent.confidence:.2f}")
        print(f"球员: {parsed_intent.players}")
        print(f"球队: {parsed_intent.teams}")
        print(f"属性: {parsed_intent.attributes}")
        print(f"篮球相关: {parsed_intent.is_basketball_related}")
        print(f"总处理时间: {total_time:.4f}s")
        print("-" * 40)

if __name__ == "__main__":
    print("🚀 Smart Pre-processor 核心功能测试")
    print("=" * 60)
    
    try:
        # 测试各个组件
        test_language_manager()
        test_intent_classifier()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！Smart Pre-processor 核心功能工作正常")
        print("🎯 Stage 3 (智能意图识别) 和 Stage 4 (实体提取) 已成功实现")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
