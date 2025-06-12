#!/usr/bin/env python3
"""
Direct Smart Pre-processor Test

直接测试smart_preprocessor.py文件中的实现
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'app', 'router'))

try:
    from smart_preprocessor import (
        UnifiedIntentClassifier, 
        NormalizedQuery,
        GlobalLanguageStateManager
    )
    
    print("✅ Successfully imported from smart_preprocessor.py")
    
    # 测试语言管理器
    print("\n🌐 测试全局语言状态管理器:")
    print("=" * 50)
    language_manager = GlobalLanguageStateManager()
    
    test_queries = [
        "How old is Kobe Bryant",
        "Compare Kobe versus Jordan", 
        "What team does LeBron play for"
    ]
    
    for query in test_queries:
        result = language_manager.detect_and_normalize_language(query)
        print(f"查询: {query}")
        print(f"语言: {result.original_language}, 标准化: {result.normalized_text}")
        print()
    
    # 测试意图分类器
    print("🧠 测试统一意图分类器:")
    print("=" * 50)
    classifier = UnifiedIntentClassifier()
    
    for query in test_queries:
        normalized = NormalizedQuery(query, query.lower(), 'en', 0.9)
        result = classifier.classify_and_extract(normalized)
        print(f"查询: {query}")
        print(f"意图: {result.intent} (置信度: {result.confidence:.2f})")
        print(f"球员: {result.players}")
        print(f"球队: {result.teams}")
        print(f"属性: {result.attributes}")
        print(f"篮球相关: {result.is_basketball_related}")
        print("-" * 30)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Cannot import from smart_preprocessor.py")
    
    # 创建一个简化的直接测试
    print("\n🔧 使用简化测试...")
    
    # 简化的测试逻辑
    queries = [
        "How old is Kobe Bryant",
        "Compare Kobe versus Jordan",
        "What team does LeBron play for"
    ]
    
    for query in queries:
        print(f"查询: {query}")
        
        # 简单的篮球检测
        basketball_keywords = ['kobe', 'jordan', 'lebron', 'basketball', 'team', 'player']
        is_basketball = any(keyword in query.lower() for keyword in basketball_keywords)
        
        print(f"篮球相关: {is_basketball}")
        print("-" * 30)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
