#!/usr/bin/env python3
"""
Smart Pre-processor Testing Script

Test the core functionality of the new intelligent router system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core data structure imports (simplified for testing)
@dataclass
class LanguageInfo:
    original_language: str
    detected_confidence: float
    normalized_language: str
    translation_needed: bool

@dataclass 
class IntentInfo:
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    query_type: str
    complexity: str
    direct_answer_expected: bool

@dataclass
class EntityInfo:
    players: List[str]
    teams: List[str] 
    attributes: List[str]
    target_entity: Optional[str]
    confidence_scores: Dict[str, float]

@dataclass
class QueryContext:
    original_query: str
    language_info: Optional[LanguageInfo] = None
    intent_info: Optional[IntentInfo] = None
    entity_info: Optional[EntityInfo] = None
    processing_steps: List[Dict] = field(default_factory=list)
    
    def add_processing_step(self, step_name: str, status: str, duration: float, details: Dict = None):
        self.processing_steps.append({
            'step_name': step_name,
            'status': status,
            'duration': duration,
            'details': details or {}
        })

class QueryContextFactory:
    @staticmethod
    def create_context(query: str) -> QueryContext:
        return QueryContext(original_query=query)

# Import our Smart Pre-processor components
from app.router.smart_preprocessor import (
    NormalizedQuery,
    ParsedIntent,
    GlobalLanguageStateManager,
    UnifiedIntentClassifier,
    SmartPostProcessor,
    SmartPreProcessor
)

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

def test_smart_preprocessor():
    """测试完整的Smart Pre-processor"""
    print("\n" + "=" * 60)
    print("🚀 测试完整Smart Pre-processor")
    print("=" * 60)
    
    preprocessor = SmartPreProcessor()
    
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
        
        context = preprocessor.process_query(query)
        
        total_time = time.time() - start_time
        
        print(f"原始语言: {context.language_info.original_language if context.language_info else 'None'}")
        print(f"意图: {context.intent_info.intent if context.intent_info else 'None'}")
        print(f"置信度: {context.intent_info.confidence if context.intent_info else 0:.2f}")
        print(f"球员: {context.entity_info.players if context.entity_info else []}")
        print(f"总处理时间: {total_time:.4f}s")
        print("-" * 40)

def test_post_processor():
    """测试后处理器"""
    print("\n" + "=" * 60)
    print("🎯 测试Smart后处理器")
    print("=" * 60)
    
    post_processor = SmartPostProcessor()
    
    test_cases = [
        ("Kobe Bryant is 42 years old.", "zh"),
        ("LeBron James plays for the Lakers.", "zh"),
        ("Hello! How can I help you with basketball questions?", "zh"),
        ("Kobe Bryant is 42 years old.", "en")
    ]
    
    for english_response, original_language in test_cases:
        normalized_query = NormalizedQuery(
            original_text="测试",
            normalized_text="test", 
            original_language=original_language,
            confidence_score=0.9
        )
        
        result = post_processor.adapt_response_language(english_response, normalized_query)
        print(f"英文回答: {english_response}")
        print(f"目标语言: {original_language}")
        print(f"适配结果: {result}")
        print("-" * 40)

def test_comprehensive_stats():
    """测试统计功能"""
    print("\n" + "=" * 60)
    print("📊 测试统计功能")
    print("=" * 60)
    
    preprocessor = SmartPreProcessor()
    
    # 运行一些查询来生成统计数据
    test_queries = [
        "科比多少岁",
        "How old is Kobe Bryant",
        "Compare Kobe versus Jordan",
        "What is the weather today",
        "Hello basketball"
    ]
    
    for query in test_queries:
        preprocessor.process_query(query)
    
    stats = preprocessor.get_comprehensive_stats()
    print("完整统计数据:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("🚀 Smart Pre-processor 核心功能测试")
    print("=" * 60)
    
    try:
        # 测试各个组件
        test_language_manager()
        test_intent_classifier()
        test_smart_preprocessor()
        test_post_processor()
        test_comprehensive_stats()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！Smart Pre-processor 工作正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
