#!/usr/bin/env python3
"""
🎯 Core模块简单功能测试 - 教学演示
演示如何使用QueryContext和core组件处理数据
不使用pytest，直接运行展示核心功能
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.schemas import (
    QueryContextFactory,
    LanguageInfo,
    EntityInfo,
    IntentInfo,
    RAGResult,
    LLMResult
)

def print_section(title: str):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """打印子节标题"""
    print(f"\n{'—'*40}")
    print(f"📋 {title}")
    print(f"{'—'*40}")

def test_basic_context_creation():
    """测试1: 基础上下文创建"""
    print_section("测试1: 基础上下文创建")
    
    # 定义我们的"钢丝线"用例
    query = "How old is Kobe Bryant?"
    print(f"🔍 测试查询: {query}")
    
    # 创建查询上下文
    context = QueryContextFactory.create(query)
    
    print(f"✅ 上下文创建成功!")
    print(f"   • 请求ID: {context.request_id}")
    print(f"   • 原始查询: {context.original_query}")
    print(f"   • 创建时间: {context.timestamp}")
    print(f"   • 状态: {context.status}")
    print(f"   • 追踪记录数: {len(context.processing_trace)}")
    
    return context

def test_language_processing(context):
    """测试2: 语言处理阶段"""
    print_section("测试2: 语言处理阶段")
    
    # 模拟语言检测和处理
    language_info = LanguageInfo(
        original_language="en",
        detected_confidence=0.95,
        normalized_language="en",
        translation_needed=False
    )
    
    # 将语言信息添加到上下文
    context.language_info = language_info
    context.normalized_query = context.original_query  # 英文无需翻译
    
    # 添加处理追踪
    context.add_trace("language_processor", "completed", {
        "original_language": language_info.original_language,
        "confidence": language_info.detected_confidence,
        "translation_needed": language_info.translation_needed
    })
    
    print(f"✅ 语言处理完成!")
    print(f"   • 检测语言: {language_info.original_language}")
    print(f"   • 检测置信度: {language_info.detected_confidence}")
    print(f"   • 需要翻译: {language_info.translation_needed}")
    print(f"   • 标准化查询: {context.normalized_query}")

def test_entity_extraction(context):
    """测试3: 实体提取阶段"""
    print_section("测试3: 实体提取阶段")
    
    # 模拟实体提取
    entity_info = EntityInfo(
        players=["Kobe Bryant"],
        attributes=["age"],
        question_words=["How", "old"],
        target_entity="Kobe Bryant"
    )
    
    # 添加置信度分数
    entity_info.confidence_scores = {
        "Kobe Bryant": 0.98,
        "age": 0.95
    }
    
    # 将实体信息添加到上下文
    context.entity_info = entity_info
    
    # 添加处理追踪
    context.add_trace("entity_extractor", "completed", {
        "players_found": len(entity_info.players),
        "attributes_found": len(entity_info.attributes),
        "target_entity": entity_info.target_entity
    })
    
    print(f"✅ 实体提取完成!")
    print(f"   • 找到球员: {entity_info.players}")
    print(f"   • 找到属性: {entity_info.attributes}")
    print(f"   • 疑问词: {entity_info.question_words}")
    print(f"   • 目标实体: {entity_info.target_entity}")
    print(f"   • 置信度分数: {entity_info.confidence_scores}")

def test_intent_classification(context):
    """测试4: 意图分类阶段"""
    print_section("测试4: 意图分类阶段")
    
    # 模拟意图分类
    intent_info = IntentInfo(
        intent="ATTRIBUTE_QUERY",
        confidence=0.92,
        query_type="simple_attribute",
        attribute_type="age",
        complexity="simple",
        direct_answer_expected=True
    )
    
    # 添加所有意图分数
    intent_info.all_scores = {
        "ATTRIBUTE_QUERY": 0.92,
        "SIMPLE_RELATION_QUERY": 0.05,
        "COMPLEX_RELATION_QUERY": 0.02,
        "DOMAIN_CHITCHAT": 0.01
    }
    
    # 将意图信息添加到上下文
    context.intent_info = intent_info
    
    # 添加处理追踪
    context.add_trace("intent_classifier", "completed", {
        "predicted_intent": intent_info.intent,
        "confidence": intent_info.confidence,
        "query_complexity": intent_info.complexity
    })
    
    print(f"✅ 意图分类完成!")
    print(f"   • 主要意图: {intent_info.intent}")
    print(f"   • 置信度: {intent_info.confidence}")
    print(f"   • 查询类型: {intent_info.query_type}")
    print(f"   • 属性类型: {intent_info.attribute_type}")
    print(f"   • 复杂度: {intent_info.complexity}")
    print(f"   • 期待直接答案: {intent_info.direct_answer_expected}")

def test_routing_stage(context):
    """测试5: 路由阶段"""
    print_section("测试5: 路由阶段")
    
    # 模拟路由决策
    context.routing_path = "direct_database"
    context.processor_selected = "DirectContextProcessor"
    context.routing_reason = "Simple attribute query with clear entity"
    context.routing_time = 0.002
    
    # 添加处理追踪
    context.add_trace("router", "completed", {
        "selected_processor": context.processor_selected,
        "routing_reason": context.routing_reason,
        "routing_time": context.routing_time
    })
    
    print(f"✅ 路由完成!")
    print(f"   • 路由路径: {context.routing_path}")
    print(f"   • 选择的处理器: {context.processor_selected}")
    print(f"   • 路由原因: {context.routing_reason}")
    print(f"   • 路由时间: {context.routing_time}s")

def test_rag_processing(context):
    """测试6: RAG处理阶段"""
    print_section("测试6: RAG处理阶段")
    
    # 模拟RAG处理结果
    rag_result = RAGResult(
        success=True,
        processor_used="DirectContextProcessor",
        processing_strategy="database_query",
        processing_time=0.156,
        context_text="Found information about Kobe Bryant:\n- Kobe Bryant, age: 41 years old, height: 198cm",
        retrieved_nodes=[
            {
                "name": "Kobe Bryant",
                "age": 41,
                "height": "198cm",
                "type": "player"
            }
        ],
        retrieved_nodes_count=1,
        confidence=0.95,
        metadata={
            "query_type": "database_lookup",
            "records_found": 1,
            "player_searched": "Kobe Bryant"
        }
    )
    
    # 将RAG结果添加到上下文
    context.rag_result = rag_result
    
    # 添加处理追踪
    context.add_trace("rag_processor", "completed", {
        "processor_used": rag_result.processor_used,
        "success": rag_result.success,
        "nodes_retrieved": len(rag_result.retrieved_nodes),
        "confidence": rag_result.confidence
    })
    
    print(f"✅ RAG处理完成!")
    print(f"   • 处理成功: {rag_result.success}")
    print(f"   • 使用的处理器: {rag_result.processor_used}")
    print(f"   • 处理策略: {rag_result.processing_strategy}")
    print(f"   • 处理时间: {rag_result.processing_time}s")
    print(f"   • 置信度: {rag_result.confidence}")
    print(f"   • 检索节点数: {len(rag_result.retrieved_nodes)}")
    print(f"   • 上下文文本: {rag_result.context_text}")

def test_llm_generation(context):
    """测试7: LLM生成阶段"""
    print_section("测试7: LLM生成阶段")
    
    # 模拟LLM生成结果
    llm_result = LLMResult(
        success=True,
        content="Kobe Bryant was 41 years old at the time of his retirement from professional basketball. He was born on August 23, 1978, and retired in 2016.",
        processing_time=1.234,
        model_used="gpt-3.5-turbo",
        tokens_used=45,
        generation_params={
            "temperature": 0.7,
            "max_tokens": 150
        },
        quality_score=0.88,
        coherence_score=0.92
    )
    
    # 将LLM结果添加到上下文
    context.llm_result = llm_result
    
    # 添加处理追踪
    context.add_trace("llm_generator", "completed", {
        "model_used": llm_result.model_used,
        "tokens_used": llm_result.tokens_used,
        "quality_score": llm_result.quality_score,
        "content_length": len(llm_result.content)
    })
    
    print(f"✅ LLM生成完成!")
    print(f"   • 生成成功: {llm_result.success}")
    print(f"   • 使用模型: {llm_result.model_used}")
    print(f"   • Token使用: {llm_result.tokens_used}")
    print(f"   • 处理时间: {llm_result.processing_time}s")
    print(f"   • 质量分数: {llm_result.quality_score}")
    print(f"   • 一致性分数: {llm_result.coherence_score}")
    print(f"   • 生成内容: {llm_result.content}")

def test_final_assembly(context):
    """测试8: 最终组装阶段"""
    print_section("测试8: 最终组装阶段")
    
    # 计算总处理时间
    context.total_processing_time = (
        context.routing_time + 
        context.rag_result.processing_time + 
        context.llm_result.processing_time + 
        0.02  # 其他处理时间
    )
    
    # 设置最终答案和状态
    context.final_answer = context.llm_result.content
    context.answer_language = "en"
    context.status = "success"
    
    # 添加质量指标
    context.quality_metrics = {
        "rag_confidence": context.rag_result.confidence,
        "llm_quality": context.llm_result.quality_score,
        "overall_confidence": (context.rag_result.confidence + context.llm_result.quality_score) / 2
    }
    
    # 标记为成功
    context.mark_success()
    
    print(f"✅ 最终组装完成!")
    print(f"   • 总处理时间: {context.total_processing_time:.3f}s")
    print(f"   • 最终状态: {context.status}")
    print(f"   • 答案语言: {context.answer_language}")
    print(f"   • 质量指标: {context.quality_metrics}")
    print(f"   • 最终答案: {context.final_answer}")

def test_context_analysis(context):
    """测试9: 上下文分析"""
    print_section("测试9: 上下文分析与总结")
    
    # 获取处理摘要
    summary = context.get_processing_summary()
    
    print_subsection("处理摘要")
    for key, value in summary.items():
        print(f"   • {key}: {value}")
    
    # 显示处理轨迹
    print_subsection("处理轨迹")
    for i, trace in enumerate(context.processing_trace, 1):
        print(f"   {i}. [{trace['component']}] {trace['action']} - {trace['timestamp'].strftime('%H:%M:%S.%f')[:-3]}")
        if trace.get('data'):
            print(f"      数据: {trace['data']}")
    
    # 显示错误和警告
    if context.errors:
        print_subsection("错误信息")
        for error in context.errors:
            print(f"   ❌ {error}")
    
    if context.warnings:
        print_subsection("警告信息")
        for warning in context.warnings:
            print(f"   ⚠️ {warning}")
    
    # API响应格式
    print_subsection("API响应格式")
    api_response = context.to_dict()
    print(f"   请求ID: {api_response['request_id']}")
    print(f"   查询: {api_response['query']}")
    print(f"   最终答案: {api_response['final_answer']}")
    print(f"   意图: {api_response['intent']}")
    print(f"   处理器: {api_response['processor_used']}")
    print(f"   状态: {api_response['status']}")

def test_mock_context_creation():
    """测试10: 使用工厂方法创建模拟上下文"""
    print_section("测试10: 工厂方法创建模拟上下文")
    
    # 使用工厂方法创建模拟上下文
    mock_context = QueryContextFactory.create_mock_context("How old is Yao Ming?")
    
    print(f"✅ 模拟上下文创建成功!")
    print(f"   • 查询: {mock_context.original_query}")
    print(f"   • 语言信息: {mock_context.language_info}")
    print(f"   • 实体信息: 球员={mock_context.entity_info.players}, 属性={mock_context.entity_info.attributes}")
    print(f"   • 意图信息: {mock_context.intent_info.intent} (置信度: {mock_context.intent_info.confidence})")
    
    # 验证上下文
    validation_errors = QueryContextFactory.validate_context(mock_context)
    if validation_errors:
        print(f"   ⚠️ 验证错误: {validation_errors}")
    else:
        print(f"   ✅ 上下文验证通过!")
    
    return mock_context

def demonstrate_data_flow():
    """核心数据流演示"""
    print_section("🎯 Core模块核心功能演示 - 数据流处理")
    
    print("""
📚 教学目标:
   1. 理解QueryContext作为系统"数据骨干"的作用
   2. 学会如何创建和操作上下文对象
   3. 了解各个处理阶段如何向上下文添加信息
   4. 掌握core模块的核心数据结构和工厂方法
   
🔧 测试用例: "How old is Kobe Bryant?"
   这是一个典型的属性查询，涵盖完整的处理流程
    """)
    
    print("\n⏳ 开始演示...")
    time.sleep(1)
    
    # 执行完整的测试流程
    context = test_basic_context_creation()
    test_language_processing(context)
    test_entity_extraction(context)
    test_intent_classification(context)
    test_routing_stage(context)
    test_rag_processing(context)
    test_llm_generation(context)
    test_final_assembly(context)
    test_context_analysis(context)
    
    # 额外演示工厂方法
    test_mock_context_creation()
    
    return context

def main():
    """主函数"""
    print("🚀 欢迎使用Core模块功能演示")
    print("这个测试将展示QueryContext的核心功能和数据流处理")
    
    try:
        # 执行核心演示
        context = demonstrate_data_flow()
        
        print_section("🎉 演示完成总结")
        print(f"""
✅ 演示成功完成！

🎯 你已经学会了:
   1. 使用QueryContextFactory.create()创建上下文
   2. 向上下文添加语言、实体、意图信息
   3. 记录RAG和LLM处理结果
   4. 使用add_trace()追踪处理过程
   5. 获取处理摘要和API响应格式
   6. 使用工厂方法创建模拟数据

📊 处理统计:
   • 总处理时间: {context.total_processing_time:.3f}秒
   • 处理阶段数: {len(context.processing_trace)}
   • 最终状态: {context.status}
   • 置信度: {context.quality_metrics.get('overall_confidence', 'N/A')}

💡 下一步可以:
   1. 尝试修改查询内容，观察处理流程的变化
   2. 学习app/rag/processors/context_aware_processor.py中的实际处理器实现
   3. 查看app/api/query_pipeline.py了解完整的流水线集成
   4. 参考demo_unified_data_flow.py获取更多演示案例
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 Core模块功能演示成功完成!")
        print("你现在应该对QueryContext的核心功能有了清晰的认识")
    else:
        print("❌ 演示未能完成，请检查错误信息")
    print(f"{'='*60}")
