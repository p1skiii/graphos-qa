#!/usr/bin/env python3
"""
优化后的三级瀑布流路由系统快速测试
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.router.intelligent_router import IntelligentRouter

def quick_test():
    """快速测试优化后的路由系统"""
    
    print("🎯 优化后三级瀑布流路由系统测试")
    print("=" * 50)
    
    router = IntelligentRouter()
    
    # 重点测试之前被误过滤的篮球查询
    basketball_test_cases = [
        "科比身高多少？",                     # 应该通过
        "湖人队有哪些球员？",                 # 应该通过
        "分析湖人队的历史成就和影响",         # 应该通过
        "詹姆斯和科比谁更强？",               # 应该通过
        "你觉得篮球怎么样？",                 # 应该通过
        "篮球规则",                          # 应该通过
        "NBA今天的比赛结果",                 # 应该通过
        
        # 非篮球查询（应该被过滤）
        "今天天气怎么样？",                   # 应该被过滤
        "帮我做数学题",                       # 应该被过滤
        "How do I cook pasta?",             # 应该被过滤
    ]
    
    basketball_passed = 0
    non_basketball_filtered = 0
    
    print("\n🔍 测试结果:")
    print("-" * 50)
    
    for i, query in enumerate(basketball_test_cases, 1):
        result = router.route_query(query)
        
        is_basketball_passed = result['processing_path'] == 'three_tier_routing'
        is_filtered = result['processing_path'] == 'zero_shot_gatekeeper_filter'
        
        if i <= 7:  # 篮球查询
            if is_basketball_passed:
                basketball_passed += 1
                print(f"✅ {query} -> {result['intent']} ({result['confidence']:.3f})")
            else:
                print(f"❌ {query} -> 被误过滤 ({result['confidence']:.3f})")
        else:  # 非篮球查询
            if is_filtered:
                non_basketball_filtered += 1
                print(f"✅ {query} -> 正确过滤 ({result['confidence']:.3f})")
            else:
                print(f"❌ {query} -> 误通过 ({result['confidence']:.3f})")
    
    print("\n📊 优化效果:")
    print("-" * 30)
    print(f"篮球查询通过率: {basketball_passed}/7 ({basketball_passed/7*100:.1f}%)")
    print(f"非篮球查询过滤率: {non_basketball_filtered}/3 ({non_basketball_filtered/3*100:.1f}%)")
    
    # 显示完整统计
    stats = router.stats
    total = stats['total_queries']
    print(f"\n总查询: {total}")
    print(f"门卫过滤: {stats['gatekeeper_filtered']} ({stats['gatekeeper_filtered']/total*100:.1f}%)")
    print(f"BERT处理: {total - stats['gatekeeper_filtered']} ({(total - stats['gatekeeper_filtered'])/total*100:.1f}%)")

if __name__ == "__main__":
    quick_test()
