#!/usr/bin/env python3
"""
测试IntentClassifier重构前后的效果对比
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.nlp.intent_classifier import IntentClassifier
from app.core.schemas import QueryContext, EntityInfo

def test_current_classifier():
    """测试当前的意图分类器"""
    print("=== 测试当前IntentClassifier效果 ===\n")
    
    classifier = IntentClassifier()
    classifier.initialize()
    
    # 测试用例
    test_cases = [
        # ATTRIBUTE_QUERY - 属性查询
        "How old is Kobe Bryant?",
        "What is LeBron James height?",
        "Which team does Stephen Curry play for?",
        
        # SIMPLE_RELATION_QUERY - 简单关系查询  
        "Did Kobe and Shaq play together?",
        "Who coached Michael Jordan?",
        
        # COMPARATIVE_QUERY - 比较查询
        "Compare Kobe Bryant and Michael Jordan",
        "Who is taller, Yao Ming or Shaq?",
        
        # DOMAIN_CHITCHAT - 篮球闲聊
        "Who is the greatest basketball player?",
        "I love basketball games",
        
        # OUT_OF_DOMAIN - 领域外
        "What's the weather today?",
        "How to cook pasta?"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"查询 {i}: {query}")
        
        # 创建简单的上下文进行测试
        context = QueryContext(
            original_query=query,
            entity_info=EntityInfo(
                players=["Kobe Bryant"] if "Kobe" in query else [],
                teams=[],
                target_entities=[]
            )
        )
        
        # 分类
        result_context = classifier.process(context)
        intent_info = result_context.intent_info
        
        print(f"  意图: {intent_info.intent}")
        print(f"  置信度: {intent_info.confidence:.2f}")
        print(f"  属性类型: {intent_info.attribute_type}")
        print(f"  复杂度: {intent_info.complexity}")
        print()

if __name__ == "__main__":
    test_current_classifier()
