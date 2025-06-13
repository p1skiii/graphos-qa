#!/usr/bin/env python3
"""
测试重构后的IntentClassifier效果
语法决策树 vs 原始硬编码规则
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.nlp.intent_classifier import IntentClassifier
from app.core.schemas import QueryContext, EntityInfo

def test_refactored_classifier():
    """测试重构后的意图分类器"""
    print("=== 测试重构后的IntentClassifier (语法决策树) ===\n")
    
    classifier = IntentClassifier()
    classifier.initialize()
    
    # 测试用例 - 增强版
    test_cases = [
        # ATTRIBUTE_QUERY - 属性查询 (单实体)
        {
            "query": "How old is Kobe Bryant?",
            "entities": {"players": ["Kobe Bryant"], "teams": []},
            "expected": "ATTRIBUTE_QUERY"
        },
        {
            "query": "What is LeBron James height?",
            "entities": {"players": ["LeBron James"], "teams": []},
            "expected": "ATTRIBUTE_QUERY"
        },
        {
            "query": "Which team does Stephen Curry play for?",
            "entities": {"players": ["Stephen Curry"], "teams": []},
            "expected": "ATTRIBUTE_QUERY"
        },
        
        # SIMPLE_RELATION_QUERY - 简单关系查询 (双实体)
        {
            "query": "Did Kobe and Shaq play together?",
            "entities": {"players": ["Kobe Bryant", "Shaq"], "teams": []},
            "expected": "SIMPLE_RELATION_QUERY"
        },
        {
            "query": "Who coached Michael Jordan?",
            "entities": {"players": ["Michael Jordan"], "teams": []},
            "expected": "SIMPLE_RELATION_QUERY"
        },
        
        # COMPARATIVE_QUERY - 比较查询 (多实体或比较词)
        {
            "query": "Compare Kobe Bryant and Michael Jordan",
            "entities": {"players": ["Kobe Bryant", "Michael Jordan"], "teams": []},
            "expected": "COMPARATIVE_QUERY"
        },
        {
            "query": "Who is taller, Yao Ming or Shaq?",
            "entities": {"players": ["Yao Ming", "Shaq"], "teams": []},
            "expected": "COMPARATIVE_QUERY"
        },
        {
            "query": "Kobe vs LeBron vs Jordan",
            "entities": {"players": ["Kobe Bryant", "LeBron James", "Michael Jordan"], "teams": []},
            "expected": "COMPARATIVE_QUERY"
        },
        
        # DOMAIN_CHITCHAT - 篮球闲聊
        {
            "query": "Who is the greatest basketball player?",
            "entities": {"players": [], "teams": []},
            "expected": "DOMAIN_CHITCHAT"
        },
        {
            "query": "I love basketball games",
            "entities": {"players": [], "teams": []},
            "expected": "DOMAIN_CHITCHAT"
        },
        {
            "query": "NBA is exciting",
            "entities": {"players": [], "teams": []},
            "expected": "DOMAIN_CHITCHAT"
        },
        
        # OUT_OF_DOMAIN - 领域外
        {
            "query": "What's the weather today?",
            "entities": {"players": [], "teams": []},
            "expected": "OUT_OF_DOMAIN"
        },
        {
            "query": "How to cook pasta?",
            "entities": {"players": [], "teams": []},
            "expected": "OUT_OF_DOMAIN"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        entities = test_case["entities"]
        expected = test_case["expected"]
        
        print(f"测试 {i}: {query}")
        
        # 创建上下文
        context = QueryContext(
            original_query=query,
            entity_info=EntityInfo(
                players=entities["players"],
                teams=entities["teams"],
                target_entities=entities["players"] + entities["teams"]
            )
        )
        
        # 分类
        result_context = classifier.process(context)
        intent_info = result_context.intent_info
        
        # 检查结果
        is_correct = intent_info.intent == expected
        if is_correct:
            correct_predictions += 1
            status = "✅ 正确"
        else:
            status = f"❌ 错误 (期望: {expected})"
        
        print(f"  意图: {intent_info.intent} {status}")
        print(f"  置信度: {intent_info.confidence:.2f}")
        print(f"  属性类型: {intent_info.attribute_type}")
        print(f"  复杂度: {intent_info.complexity}")
        print(f"  查询类型: {intent_info.query_type}")
        print()
    
    # 总结
    accuracy = correct_predictions / total_tests
    print(f"=== 语法决策树分类器性能总结 ===")
    print(f"总测试数: {total_tests}")
    print(f"正确预测: {correct_predictions}")
    print(f"准确率: {accuracy:.2%}")
    print(f"代码行数: ~100行 (vs 原来300+行)")
    print("决策方法: 语法决策树 (vs 硬编码规则)")

if __name__ == "__main__":
    test_refactored_classifier()
