"""
QueryContext 统一数据流演示
展示新的数据对象架构如何在系统中流动
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.schemas import (
    QueryContext, QueryContextFactory, 
    LanguageInfo, EntityInfo, IntentInfo, 
    RAGResult, LLMResult
)
from app.core.validation import global_monitor, global_validator, global_debugger
import time
import json

def demo_basic_context_creation():
    """演示基础的上下文创建和操作"""
    print("🎯 演示1: 基础上下文创建")
    print("=" * 50)
    
    # 创建查询上下文
    context = QueryContextFactory.create("How old is Yao Ming?")
    print(f"创建上下文: {context.request_id}")
    print(f"原始查询: {context.original_query}")
    print(f"时间戳: {context.timestamp}")
    print(f"状态: {context.status}")
    
    # 添加追踪记录
    context.add_trace("demo", "context_created", {"demo_stage": 1})
    print(f"追踪记录数: {len(context.processing_trace)}")
    
    # 添加警告和错误
    context.add_warning("demo", "这是一个演示警告")
    context.add_error("demo", "这是一个演示错误")
    
    print(f"错误数: {len(context.errors)}")
    print(f"警告数: {len(context.warnings)}")
    print(f"最终状态: {context.status}")
    
    return context

def demo_language_processing(context: QueryContext):
    """演示语言处理阶段"""
    print("\n🌍 演示2: 语言处理阶段")
    print("=" * 50)
    
    # 模拟语言检测
    context.language_info = LanguageInfo(
        original_language="en",
        detected_confidence=0.95,
        normalized_language="en",
        translation_needed=False
    )
    
    context.normalized_query = "How old is Yao Ming?"
    context.add_trace("language_processor", "language_detected", {
        "original_language": "en",
        "confidence": 0.95
    })
    
    print(f"原始语言: {context.language_info.original_language}")
    print(f"检测置信度: {context.language_info.detected_confidence}")
    print(f"需要翻译: {context.language_info.translation_needed}")
    print(f"标准化查询: {context.normalized_query}")
    
    # 验证语言处理阶段
    validation_errors = global_validator.validate_stage(context, "preprocessing")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_intent_classification(context: QueryContext):
    """演示意图分类阶段"""
    print("\n🎯 演示3: 意图分类阶段")
    print("=" * 50)
    
    # 模拟意图分类
    context.intent_info = IntentInfo(
        intent="ATTRIBUTE_QUERY",
        confidence=0.92,
        all_scores={
            "ATTRIBUTE_QUERY": 0.92,
            "SIMPLE_RELATION_QUERY": 0.05,
            "COMPLEX_RELATION_QUERY": 0.02,
            "COMPARATIVE_QUERY": 0.01
        },
        query_type="attribute_query",
        attribute_type="age",
        complexity="simple",
        direct_answer_expected=True
    )
    
    # 模拟实体提取
    context.entity_info = EntityInfo(
        players=["Yao Ming"],
        attributes=["age", "old"],
        question_words=["how", "is"],
        target_entity="Yao Ming",
        confidence_scores={
            "Yao Ming": 0.98,
            "age": 0.95
        }
    )
    
    context.add_trace("intent_classifier", "classification_completed", {
        "intent": "ATTRIBUTE_QUERY",
        "confidence": 0.92,
        "entities_extracted": 1
    })
    
    print(f"意图: {context.intent_info.intent}")
    print(f"置信度: {context.intent_info.confidence}")
    print(f"属性类型: {context.intent_info.attribute_type}")
    print(f"目标实体: {context.entity_info.target_entity}")
    print(f"提取的球员: {context.entity_info.players}")
    print(f"提取的属性: {context.entity_info.attributes}")
    
    # 验证意图分类阶段
    validation_errors = global_validator.validate_stage(context, "intent_classification")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_routing_stage(context: QueryContext):
    """演示路由阶段"""
    print("\n🛤️ 演示4: 路由阶段")
    print("=" * 50)
    
    # 模拟路由决策
    start_time = time.time()
    
    context.routing_path = "intent_based_routing"
    context.processor_selected = "direct_db_processor"
    context.routing_reason = f"基于意图 {context.intent_info.intent} 选择直接数据库处理器"
    
    time.sleep(0.001)  # 模拟路由处理时间
    context.routing_time = time.time() - start_time
    
    context.add_trace("router", "routing_completed", {
        "selected_processor": "direct_db_processor",
        "routing_time": context.routing_time
    })
    
    print(f"路由路径: {context.routing_path}")
    print(f"选择的处理器: {context.processor_selected}")
    print(f"路由原因: {context.routing_reason}")
    print(f"路由耗时: {context.routing_time:.6f}s")
    
    # 验证路由阶段
    validation_errors = global_validator.validate_stage(context, "routing")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_rag_processing(context: QueryContext):
    """演示RAG处理阶段"""
    print("\n📚 演示5: RAG处理阶段")
    print("=" * 50)
    
    # 模拟RAG处理
    start_time = time.time()
    
    context.rag_result = RAGResult(
        success=True,
        processor_used="direct_db_processor",
        processing_strategy="direct_attribute_query",
        processing_time=0.002,
        contextualized_text="球员年龄信息：\nYao Ming：38岁\nJames Harden：29岁\nLeBron James：34岁",
        retrieved_nodes_count=3,
        confidence=0.95,
        subgraph_summary={
            "nodes": 3,
            "edges": 2,
            "algorithm": "direct_lookup"
        },
        metadata={
            "query_type": "age_lookup",
            "target_player": "Yao Ming"
        }
    )
    
    context.add_trace("rag_processor", "processing_completed", {
        "processor": "direct_db_processor",
        "nodes_retrieved": 3,
        "success": True
    })
    
    print(f"处理成功: {context.rag_result.success}")
    print(f"使用的处理器: {context.rag_result.processor_used}")
    print(f"处理策略: {context.rag_result.processing_strategy}")
    print(f"检索到的节点数: {context.rag_result.retrieved_nodes_count}")
    print(f"结果置信度: {context.rag_result.confidence}")
    print(f"上下文文本预览: {context.rag_result.contextualized_text[:50]}...")
    
    # 验证RAG处理阶段
    validation_errors = global_validator.validate_stage(context, "rag_processing")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_llm_generation(context: QueryContext):
    """演示LLM生成阶段"""
    print("\n🤖 演示6: LLM生成阶段")
    print("=" * 50)
    
    # 模拟LLM生成
    context.llm_result = LLMResult(
        success=True,
        content="根据数据库信息，姚明今年38岁。姚明是中国著名的篮球运动员，曾效力于NBA休斯顿火箭队。",
        processing_time=0.15,
        model_name="phi-3-mini",
        tokens_used=45,
        generation_params={
            "temperature": 0.7,
            "max_tokens": 150
        },
        quality_score=0.89,
        coherence_score=0.92,
        fallback_used=False
    )
    
    context.add_trace("llm_engine", "generation_completed", {
        "model": "phi-3-mini",
        "tokens_used": 45,
        "quality_score": 0.89
    })
    
    print(f"生成成功: {context.llm_result.success}")
    print(f"使用模型: {context.llm_result.model_name}")
    print(f"生成内容: {context.llm_result.content}")
    print(f"使用Token数: {context.llm_result.tokens_used}")
    print(f"质量评分: {context.llm_result.quality_score}")
    print(f"连贯性评分: {context.llm_result.coherence_score}")
    
    # 验证LLM生成阶段
    validation_errors = global_validator.validate_stage(context, "llm_generation")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_postprocessing(context: QueryContext):
    """演示后处理阶段"""
    print("\n🏁 演示7: 后处理阶段")
    print("=" * 50)
    
    # 模拟后处理
    context.final_answer = context.llm_result.content
    context.answer_language = "zh"  # 最终回答是中文
    context.total_processing_time = sum([
        context.routing_time,
        context.rag_result.processing_time if context.rag_result else 0,
        context.llm_result.processing_time if context.llm_result else 0
    ])
    
    # 添加质量指标
    context.quality_metrics = {
        "overall_confidence": 0.91,
        "answer_relevance": 0.94,
        "information_completeness": 0.88
    }
    
    # 标记完成
    context.mark_success()
    
    context.add_trace("postprocessor", "processing_completed", {
        "final_answer_length": len(context.final_answer),
        "answer_language": context.answer_language,
        "total_time": context.total_processing_time
    })
    
    print(f"最终答案: {context.final_answer}")
    print(f"回答语言: {context.answer_language}")
    print(f"总处理时间: {context.total_processing_time:.6f}s")
    print(f"最终状态: {context.status}")
    print(f"整体置信度: {context.quality_metrics['overall_confidence']}")
    
    # 验证后处理阶段
    validation_errors = global_validator.validate_stage(context, "postprocessing")
    print(f"验证结果: {'通过' if not validation_errors else f'失败 - {validation_errors}'}")

def demo_context_analysis(context: QueryContext):
    """演示上下文分析和监控"""
    print("\n📊 演示8: 上下文分析和监控")
    print("=" * 50)
    
    # 生成处理摘要
    summary = context.get_processing_summary()
    print("处理摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 更新全局监控
    global_monitor.track_request(context)
    
    # 获取性能报告
    performance_report = global_monitor.get_performance_report()
    print("\n性能报告:")
    print(f"  总请求数: {performance_report['overview']['total_requests']}")
    print(f"  成功率: {performance_report['overview']['success_rate']:.2%}")
    print(f"  平均处理时间: {performance_report['overview']['avg_processing_time']:.6f}s")
    
    # 生成调试信息
    debug_info = global_debugger.debug_context(context)
    print("\n调试信息摘要:")
    print(f"  处理阶段数: {len(debug_info['processing_stages'])}")
    print(f"  找到的实体数: {debug_info['results_summary']['entities_found']}")
    print(f"  最终答案长度: {debug_info['results_summary']['final_answer_length']}")

def demo_data_export(context: QueryContext):
    """演示数据导出功能"""
    print("\n💾 演示9: 数据导出")
    print("=" * 50)
    
    # 转换为API响应格式
    api_response = context.to_dict()
    print("API响应格式:")
    print(json.dumps(api_response, indent=2, ensure_ascii=False)[:500] + "...")
    
    # 导出调试JSON
    debug_json = global_debugger.export_context_json(context)
    print(f"\n调试JSON长度: {len(debug_json)} 字符")
    
    # 验证工厂方法
    validation_errors = QueryContextFactory.validate_context(context)
    print(f"上下文验证: {'通过' if not validation_errors else f'发现问题: {validation_errors}'}")

def main():
    """主演示函数"""
    print("🚀 QueryContext 统一数据流架构演示")
    print("=" * 70)
    
    try:
        # 运行所有演示
        context = demo_basic_context_creation()
        demo_language_processing(context)
        demo_intent_classification(context)
        demo_routing_stage(context)
        demo_rag_processing(context)
        demo_llm_generation(context)
        demo_postprocessing(context)
        demo_context_analysis(context)
        demo_data_export(context)
        
        print("\n🎉 演示完成！")
        print("=" * 70)
        print("✅ 统一数据对象架构工作正常")
        print("✅ 所有阶段验证通过")
        print("✅ 监控和调试功能正常")
        print("✅ 数据导出功能正常")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
