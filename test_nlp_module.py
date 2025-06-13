#!/usr/bin/env python
"""
NLP模块功能测试
测试刚实现的NLP组件功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from app.core.schemas import QueryContextFactory
from app.services.nlp import NLPPipeline, create_default_nlp_pipeline

def test_individual_components():
    """测试各个NLP组件的独立功能"""
    print("🧪 测试NLP组件独立功能...")
    
    # 创建管道
    pipeline = create_default_nlp_pipeline()
    
    # 测试查询
    test_queries = [
        "How old is Kobe Bryant?",
        "What position does LeBron James play?", 
        "Who is taller, Michael Jordan or Yao Ming?",
        "I love basketball"
    ]
    
    for query in test_queries:
        print(f"\n📝 测试查询: '{query}'")
        print("-" * 60)
        
        try:
            # 测试各组件
            component_results = pipeline.test_individual_components(query)
            
            # 显示结果
            for component, result in component_results.items():
                print(f"🔧 {component}:")
                if "error" in result:
                    print(f"   ❌ 错误: {result['error']}")
                else:
                    print(f"   ✅ 结果: {result}")
                    
        except Exception as e:
            print(f"❌ 测试失败: {e}")

def test_full_pipeline():
    """测试完整的NLP管道"""
    print("\n" + "="*80)
    print("🔄 测试完整NLP管道...")
    
    # 创建管道
    pipeline = create_default_nlp_pipeline()
    
    # 初始化
    if not pipeline.initialize():
        print("❌ NLP管道初始化失败")
        return
    
    print("✅ NLP管道初始化成功")
    
    # 测试查询
    test_queries = [
        "How old is Kobe Bryant?",
        "What team does LeBron James play for?",
        "Who is the greatest basketball player?",
        "Compare Michael Jordan and Kobe Bryant"
    ]
    
    for query in test_queries:
        print(f"\n📝 处理查询: '{query}'")
        print("-" * 60)
        
        try:
            # 完整处理
            result_context = pipeline.process_query(query)
            
            # 显示结果
            print(f"🔍 语言检测: {result_context.language_info.original_language if result_context.language_info else 'None'}")
            print(f"🔤 分词数量: {len(result_context.tokens) if hasattr(result_context, 'tokens') else 0}")
            
            if result_context.entity_info:
                print(f"🏀 实体提取:")
                print(f"   - 球员: {result_context.entity_info.players}")
                print(f"   - 属性: {result_context.entity_info.attributes}")
                print(f"   - 目标实体: {result_context.entity_info.target_entity}")
            
            if result_context.intent_info:
                print(f"🎯 意图分类:")
                print(f"   - 意图: {result_context.intent_info.intent}")
                print(f"   - 置信度: {result_context.intent_info.confidence:.2f}")
                print(f"   - 属性类型: {result_context.intent_info.attribute_type}")
                print(f"   - 复杂度: {result_context.intent_info.complexity}")
            
            # 显示处理轨迹
            print(f"📊 处理轨迹: {len(result_context.processing_trace)} 步骤")
            for trace in result_context.processing_trace[-3:]:  # 显示最后3个步骤
                print(f"   - {trace['component']}: {trace['action']}")
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

def test_core_integration():
    """测试与core模块的集成"""
    print("\n" + "="*80)
    print("🔗 测试与Core模块的集成...")
    
    # 创建QueryContext
    context = QueryContextFactory.create("How old is Kobe Bryant?")
    print(f"✅ 创建QueryContext: {context.request_id[:8]}...")
    
    # 创建NLP管道
    pipeline = create_default_nlp_pipeline()
    
    if not pipeline.initialize():
        print("❌ 初始化失败")
        return
    
    # 处理
    try:
        result = pipeline.process(context)
        
        print(f"🎯 处理结果:")
        print(f"   - 原始查询: {result.original_query}")
        print(f"   - 语言: {result.language_info.original_language if result.language_info else 'unknown'}")
        print(f"   - 意图: {result.intent_info.intent if result.intent_info else 'unknown'}")
        print(f"   - 状态: {result.status}")
        print(f"   - 处理时间: {result.total_processing_time}")
        
        # 验证数据完整性
        assert result.request_id == context.request_id, "RequestID不匹配"
        assert result.original_query == context.original_query, "原始查询不匹配"
        
        print("✅ Core集成测试通过")
        
    except Exception as e:
        print(f"❌ Core集成测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 NLP模块测试开始")
    print("="*80)
    
    # 依赖检查
    try:
        import langdetect
        import spacy
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请安装必要依赖:")
        print("pip install langdetect spacy")
        print("python -m spacy download en_core_web_sm")
        return
    
    # 运行测试
    try:
        test_individual_components()
        test_full_pipeline()
        test_core_integration()
        
        print("\n" + "="*80)
        print("🎉 NLP模块测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
