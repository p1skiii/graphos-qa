#!/usr/bin/env python3
"""
现代化三级瀑布流路由系统测试
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到path
sys.path.append(str(Path(__file__).parent))

from app.router.intelligent_router import IntelligentRouter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_three_tier_routing():
    """测试三级瀑布流路由系统"""
    
    print("🏀 现代化三级瀑布流路由系统测试")
    print("=" * 60)
    
    # 初始化路由器
    router = IntelligentRouter()
    
    # 测试用例
    test_cases = [
        # 第一级：Zero-shot门卫过滤测试
        "今天天气怎么样？",                    # 应该被门卫过滤
        "What's the weather like today?",    # 应该被门卫过滤
        "帮我做数学题",                       # 应该被门卫过滤
        "How do I cook pasta?",             # 应该被门卫过滤
        
        # 第二级：BERT 5类分类测试
        "科比身高多少？",                     # ATTRIBUTE_QUERY
        "How tall is Kobe Bryant?",         # ATTRIBUTE_QUERY
        "湖人队有哪些球员？",                 # SIMPLE_RELATION_QUERY
        "Who plays for the Lakers?",       # SIMPLE_RELATION_QUERY
        "分析湖人队的历史成就和影响",         # COMPLEX_RELATION_QUERY
        "Analyze Lakers' historical achievements",  # COMPLEX_RELATION_QUERY
        "詹姆斯和科比谁更强？",               # COMPARATIVE_QUERY
        "Who is better, LeBron or Kobe?",   # COMPARATIVE_QUERY
        "你觉得篮球怎么样？",                 # DOMAIN_CHITCHAT
        "What do you think about basketball?",  # DOMAIN_CHITCHAT
        
        # 边界测试
        "篮球规则",                          # 简单篮球相关
        "Basketball rules",                 # 简单篮球相关
        "NBA今天的比赛结果",                 # 篮球相关但可能无具体数据
        "Today's NBA game results"          # 篮球相关但可能无具体数据
    ]
    
    results = []
    
    print("\n🔍 开始路由测试...")
    print("-" * 60)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n查询 {i}: \"{query}\"")
        try:
            result = router.route_query(query)
            results.append(result)
            
            # 显示路由结果
            print(f"  ✓ 意图: {result['intent']}")
            print(f"  ✓ 置信度: {result['confidence']:.3f}")
            print(f"  ✓ 处理器: {result['processor']}")
            print(f"  ✓ 路径: {result['processing_path']}")
            print(f"  ✓ 原因: {result['reason']}")
            print(f"  ⏱️ 门卫时间: {result['gatekeeper_time']*1000:.2f}ms")
            print(f"  ⏱️ BERT时间: {result['bert_time']*1000:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 显示统计信息
    print("\n📊 路由统计信息")
    print("=" * 60)
    stats = router.stats
    
    print(f"总查询数: {stats['total_queries']}")
    print(f"门卫过滤数: {stats['gatekeeper_filtered']}")
    print(f"属性查询: {stats['attribute_queries']}")
    print(f"简单关系查询: {stats['simple_relation_queries']}")
    print(f"复杂关系查询: {stats['complex_relation_queries']}")
    print(f"比较查询: {stats['comparative_queries']}")
    print(f"领域闲聊: {stats['domain_chitchat']}")
    print(f"平均处理时间: {stats['avg_processing_time']*1000:.2f}ms")
    print(f"平均门卫时间: {stats['gatekeeper_time']*1000:.2f}ms")
    print(f"平均BERT时间: {stats['bert_classification_time']*1000:.2f}ms")
    
    # 分析路由分布
    print("\n📈 路由分布分析")
    print("-" * 30)
    
    intent_counts = {}
    for result in results:
        intent = result['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in sorted(intent_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"{intent}: {count} ({percentage:.1f}%)")
    
    print("\n✅ 三级瀑布流路由系统测试完成!")
    
    return results

if __name__ == "__main__":
    test_three_tier_routing()
