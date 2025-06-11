"""
Phase 1 LLM集成演示脚本
展示基础LLM框架的核心功能和处理流程
"""
import sys
import os
import time
import json

# 添加项目路径
sys.path.append('/Users/wang/i/graphos-qa')

try:
    from app.llm import (
        create_llm_system, get_available_presets,
        UnifiedInput, LLMResponse
    )
    print("✅ 成功导入LLM模块")
except ImportError as e:
    print(f"❌ LLM模块导入失败: {e}")
    sys.exit(1)

def demo_system_creation():
    """演示系统创建和初始化"""
    print("🎯 演示1: LLM系统创建和初始化")
    print("="*50)
    
    # 显示可用预设
    presets = get_available_presets()
    print(f"📋 可用预设: {presets}")
    
    # 创建MacOS优化系统
    print(f"\n🔧 创建MacOS优化LLM系统...")
    system = create_llm_system('macos_optimized')
    
    print(f"✅ 系统创建完成")
    print(f"   配置类型: macos_optimized")
    print(f"   模型: {system.config.llm_config.model_config.model_name}")
    print(f"   设备: {system.config.llm_config.model_config.device}")
    print(f"   多模态支持: {system.config.llm_config.enable_multimodal}")
    
    # 初始化系统
    print(f"\n🔄 初始化系统组件...")
    if system.initialize():
        print(f"✅ 系统初始化成功")
        
        # 显示组件状态
        status = system.get_system_status()
        print(f"   初始化状态: {status['initialized']}")
        print(f"   就绪状态: {status['ready']}")
        print(f"   组件状态:")
        for component, loaded in status['components'].items():
            print(f"     {component}: {'✅' if loaded else '❌'}")
    else:
        print(f"❌ 系统初始化失败")
        return None
    
    return system

def demo_input_processing():
    """演示输入处理流程"""
    print("\n🎯 演示2: 输入处理流程")
    print("="*50)
    
    # 创建系统
    system = create_llm_system('development')
    system.initialize()
    
    # 模拟不同类型的RAG处理器输出
    test_cases = [
        {
            'name': 'Direct处理器 - 简单问答',
            'query': '科比的身高是多少？',
            'processor_output': {
                'textual_context': {
                    'formatted_text': '科比·布莱恩特身高6英尺6英寸（198厘米），是一名得分后卫。'
                },
                'metadata': {'processor_name': 'direct_processor', 'confidence': 0.95},
                'success': True
            }
        },
        {
            'name': 'Simple G处理器 - 图谱增强',
            'query': '科比和沙克的关系如何？',
            'processor_output': {
                'textual_context': {
                    'formatted_text': '科比·布莱恩特和沙奇尔·奥尼尔在洛杉矶湖人队是队友，他们一起获得了三连冠（2000-2002）。'
                },
                'graph': {
                    'nodes': [
                        {'id': 'kobe', 'name': '科比', 'type': 'player'},
                        {'id': 'shaq', 'name': '沙克', 'type': 'player'},
                        {'id': 'lakers', 'name': '湖人', 'type': 'team'}
                    ],
                    'edges': [
                        {'source': 'kobe', 'target': 'lakers', 'relation': 'plays_for'},
                        {'source': 'shaq', 'target': 'lakers', 'relation': 'plays_for'},
                        {'source': 'kobe', 'target': 'shaq', 'relation': 'teammate'}
                    ]
                },
                'metadata': {'processor_name': 'simple_g_processor'},
                'success': True
            }
        },
        {
            'name': 'Complex G处理器 - 多模态增强',
            'query': '分析科比和詹姆斯的职业成就对比',
            'processor_output': {
                'mode': 'enhanced',
                'traditional_result': {
                    'textual_context': {
                        'formatted_text': '科比获得5次总冠军、18次全明星，詹姆斯获得4次总冠军、19次全明星。两人都是NBA历史上的传奇球员。'
                    }
                },
                'graph_embedding': {
                    'embedding': [0.23, -0.15, 0.67, 0.42, -0.31, 0.89, -0.56, 0.78, 0.12, -0.44],
                    'encoding_success': True
                },
                'multimodal_context': None,  # 在实际应用中会有MultimodalContext对象
                'metadata': {'processor_name': 'complex_g_processor'},
                'success': True
            }
        },
        {
            'name': 'Comparison处理器 - 比较分析',
            'query': '比较科比和乔丹的得分能力',
            'processor_output': {
                'comparison_result': {
                    'formatted_text': '科比职业生涯总得分33,643分，场均25.0分；乔丹职业生涯总得分32,292分，场均30.1分。乔丹的场均得分更高，但科比的总得分更多。'
                },
                'comparison_subjects': ['科比', '乔丹'],
                'comparison_aspects': ['得分能力', '效率', '职业生涯长度'],
                'metadata': {'processor_name': 'comparison_processor'},
                'success': True
            }
        },
        {
            'name': 'Chitchat处理器 - 闲聊互动',
            'query': '你觉得篮球比赛最精彩的是什么？',
            'processor_output': {
                'response': '篮球比赛最精彩的是那些关键时刻的绝杀，还有球员之间的精妙配合。',
                'metadata': {'processor_name': 'chitchat_processor'},
                'success': True
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   📝 案例 {i}: {test_case['name']}")
        print(f"   查询: {test_case['query']}")
        
        # 输入路由
        unified_input = system.input_router.route_processor_output(
            test_case['processor_output'], 
            test_case['query']
        )
        
        print(f"   ✅ 路由完成: {unified_input.processor_type}")
        print(f"   多模态数据: {'有' if unified_input.has_multimodal_data() else '无'}")
        print(f"   文本内容长度: {len(unified_input.get_text_content())}")
        
        # Prompt生成
        prompt = system.prompt_manager.format_prompt_for_input(
            unified_input, 
            "你是一个专业的篮球知识问答助手"
        )
        print(f"   ✅ Prompt生成完成，长度: {len(prompt)}")
    
    # 显示路由器统计
    stats = system.input_router.get_stats()
    print(f"\n📊 输入路由器统计:")
    print(f"   总处理数: {stats['total_processed']}")
    print(f"   处理器分布: {stats['processor_distribution']}")
    print(f"   多模态比例: {stats['multimodal_ratio']:.2%}")

def demo_prompt_templates():
    """演示Prompt模板系统"""
    print("\n🎯 演示3: Prompt模板系统")
    print("="*50)
    
    # 创建系统
    system = create_llm_system('development')
    system.initialize()
    
    # 获取模板管理器
    manager = system.prompt_manager
    
    print(f"📝 模板管理器信息:")
    print(f"   总模板数: {len(manager.templates)}")
    print(f"   可用模板: {manager.list_templates()}")
    
    # 展示每种模板的详细信息
    template_names = ['simple_qa', 'multimodal_qa', 'comparison_qa', 'chitchat']
    
    for template_name in template_names:
        print(f"\n   📋 模板: {template_name}")
        info = manager.get_template_info(template_name)
        if info:
            print(f"      类型: {info['type']}")
            print(f"      必需字段: {info['required_fields']}")
            print(f"      可选字段: {info['optional_fields']}")
            print(f"      描述: {info['description']}")
            print(f"      有效性: {'✅' if info['is_valid'] else '❌'}")
    
    # 演示自定义Prompt生成
    print(f"\n🔧 自定义Prompt生成演示:")
    
    # 创建测试输入
    unified_input = UnifiedInput(
        query="科比获得过哪些荣誉？",
        processor_type="complex_g",
        text_context="科比在NBA生涯中获得了众多荣誉，包括总冠军、全明星、MVP等。",
        graph_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    # 生成多模态Prompt
    prompt = manager.format_prompt_for_input(
        unified_input, 
        "你是NBA历史专家，擅长分析球员成就"
    )
    
    print(f"   ✅ 多模态Prompt生成成功")
    print(f"   长度: {len(prompt)}")
    print(f"   预览:")
    print("   " + "-" * 40)
    print("   " + prompt[:200].replace('\n', '\n   ') + "...")
    print("   " + "-" * 40)

def demo_response_formatting():
    """演示响应格式化"""
    print("\n🎯 演示4: 响应格式化")
    print("="*50)
    
    # 创建系统
    system = create_llm_system('development')
    system.initialize()
    
    # 模拟不同类型的LLM响应
    test_responses = [
        {
            'name': '简单问答响应',
            'llm_response': LLMResponse(
                content="科比·布莱恩特身高6英尺6英寸，约198厘米。他是一名得分后卫，在NBA生涯中以其出色的得分能力而闻名。",
                metadata={'model': 'phi-3-mini', 'temperature': 0.7},
                processing_time=1.2,
                token_usage={'input_tokens': 45, 'output_tokens': 35, 'total_tokens': 80}
            ),
            'unified_input': UnifiedInput(
                query="科比的身高是多少？",
                processor_type="direct",
                text_context="科比基本信息"
            )
        },
        {
            'name': '比较分析响应',
            'llm_response': LLMResponse(
                content="科比和乔丹在得分方面各有特色。1. 场均得分：乔丹30.1分，科比25.0分。2. 总得分：科比33,643分，乔丹32,292分。3. 得分效率：乔丹更高。4. 职业生涯长度：科比更长。",
                metadata={'model': 'phi-3-mini', 'temperature': 0.7},
                processing_time=2.8,
                token_usage={'input_tokens': 120, 'output_tokens': 85, 'total_tokens': 205}
            ),
            'unified_input': UnifiedInput(
                query="比较科比和乔丹的得分能力",
                processor_type="comparison",
                text_context="科比和乔丹的得分数据比较"
            )
        },
        {
            'name': '多模态分析响应',
            'llm_response': LLMResponse(
                content="基于图谱分析和文本信息，科比和詹姆斯都是NBA历史上的传奇球员。科比在湖人队获得5次总冠军，詹姆斯则在多支球队都取得了成功。他们的关系从早期的竞争发展为后来的相互尊重。",
                metadata={'model': 'phi-3-mini', 'temperature': 0.7, 'multimodal': True},
                processing_time=3.5,
                token_usage={'input_tokens': 180, 'output_tokens': 120, 'total_tokens': 300}
            ),
            'unified_input': UnifiedInput(
                query="分析科比和詹姆斯的关系",
                processor_type="complex_g",
                text_context="科比和詹姆斯的职业生涯分析",
                graph_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
        }
    ]
    
    for i, test_case in enumerate(test_responses, 1):
        print(f"\n   📝 案例 {i}: {test_case['name']}")
        
        # 格式化响应
        formatted_response = system.response_formatter.format_response(
            test_case['llm_response'],
            test_case['unified_input']
        )
        
        print(f"   ✅ 格式化完成")
        print(f"   格式类型: {formatted_response.format_type}")
        print(f"   原始长度: {len(test_case['llm_response'].content)}")
        print(f"   格式化后长度: {len(formatted_response.content)}")
        print(f"   处理时间: {formatted_response.processing_info['formatting_time']:.3f}秒")
        
        # 显示格式化后的内容
        print(f"   内容预览:")
        preview = formatted_response.content[:150].replace('\n', '\n      ')
        print(f"      {preview}...")
    
    # 显示格式化器统计
    stats = system.response_formatter.get_stats()
    print(f"\n📊 响应格式化器统计:")
    print(f"   总格式化数: {stats['total_formatted']}")
    print(f"   平均处理时间: {stats['performance']['avg_formatting_time']:.4f}秒")
    print(f"   截断率: {stats['quality_metrics']['truncation_rate']:.2%}")

def demo_configuration_options():
    """演示配置选项"""
    print("\n🎯 演示5: 配置选项和自定义")
    print("="*50)
    
    from app.llm.factory import llm_factory
    
    # 显示所有预设配置
    presets = llm_factory.list_presets()
    print(f"📋 可用预设配置:")
    
    for preset in presets:
        info = llm_factory.get_preset_info(preset)
        if info:
            print(f"   🔧 {preset}:")
            print(f"      模型: {info['model_name']}")
            print(f"      设备: {info['device']}")
            print(f"      多模态: {info['multimodal_enabled']}")
            print(f"      自动加载: {info['auto_load_model']}")
            print(f"      描述: {info['description']}")
    
    # 演示自定义配置
    print(f"\n🎛️ 自定义配置演示:")
    
    custom_config = {
        'llm_config': {
            'max_input_tokens': 2048,
            'temperature': 0.5,
            'system_prompt': '你是一个专业的NBA分析师'
        },
        'formatter_config': {
            'enable_markdown': True,
            'highlight_keywords': True,
            'max_response_length': 1024
        },
        'auto_load_model': False
    }
    
    print(f"   自定义配置:")
    print(f"   {json.dumps(custom_config, indent=6, ensure_ascii=False)}")
    
    # 创建自定义系统
    custom_system = llm_factory.create_system('macos_optimized', custom_config)
    custom_system.initialize()
    
    print(f"   ✅ 自定义系统创建成功")
    print(f"   最大输入tokens: {custom_system.config.llm_config.max_input_tokens}")
    print(f"   系统prompt: {custom_system.config.llm_config.system_prompt}")

def demo_system_status():
    """演示系统状态监控"""
    print("\n🎯 演示6: 系统状态监控")
    print("="*50)
    
    # 创建系统
    system = create_llm_system('macos_optimized')
    system.initialize()
    
    # 获取完整系统状态
    status = system.get_system_status()
    
    print(f"🖥️ 系统整体状态:")
    print(f"   初始化: {'✅' if status['initialized'] else '❌'}")
    print(f"   就绪: {'✅' if status['ready'] else '❌'}")
    
    print(f"\n🔧 组件状态:")
    for component, loaded in status['components'].items():
        print(f"   {component}: {'✅' if loaded else '❌'}")
    
    # 显示各组件详细统计
    if 'engine_stats' in status:
        engine_stats = status['engine_stats']
        print(f"\n🤖 LLM引擎状态:")
        print(f"   模型加载: {'✅' if engine_stats['model_loaded'] else '❌'}")
        print(f"   模型名称: {engine_stats['model_name']}")
        print(f"   设备: {engine_stats['device']}")
        print(f"   多模态启用: {'✅' if engine_stats['multimodal_enabled'] else '❌'}")
        print(f"   图投影器启用: {'✅' if engine_stats['graph_projector_enabled'] else '❌'}")
    
    if 'router_stats' in status:
        router_stats = status['router_stats']
        print(f"\n🚦 输入路由器状态:")
        print(f"   总处理数: {router_stats['total_processed']}")
        print(f"   支持处理器: {router_stats['supported_processors']}")
    
    if 'formatter_stats' in status:
        formatter_stats = status['formatter_stats']
        print(f"\n📝 响应格式化器状态:")
        print(f"   总格式化数: {formatter_stats['total_formatted']}")
        print(f"   配置: {formatter_stats['configuration']}")

def main():
    """主演示函数"""
    print("🚀 Phase 1 LLM集成演示")
    print("基于G-Retriever论文的三阶段LLM集成方案")
    print("=" * 60)
    print()
    
    try:
        # 演示1: 系统创建
        system = demo_system_creation()
        if not system:
            print("❌ 系统创建失败，终止演示")
            return
        
        # 演示2: 输入处理
        demo_input_processing()
        
        # 演示3: Prompt模板
        demo_prompt_templates()
        
        # 演示4: 响应格式化
        demo_response_formatting()
        
        # 演示5: 配置选项
        demo_configuration_options()
        
        # 演示6: 系统状态
        demo_system_status()
        
        print("\n" + "=" * 60)
        print("🎉 Phase 1 LLM集成演示完成！")
        print()
        print("📋 Phase 1 核心功能总结:")
        print("   ✅ 统一LLM架构 - 支持Phi-3-mini模型")
        print("   ✅ 统一输入接口 - 处理5种RAG处理器输出")
        print("   ✅ Prompt模板系统 - 适配不同查询类型")
        print("   ✅ 响应格式化 - 统一输出格式")
        print("   ✅ 配置管理 - 多种预设和自定义选项")
        print("   ✅ 工厂模式 - 便捷的组件创建")
        print("   ✅ 状态监控 - 完整的系统状态跟踪")
        print()
        print("🎯 Phase 1 目标达成:")
        print("   ✅ 搭建了Phi-3-mini基础框架")
        print("   ✅ 实现了统一输入接口")
        print("   ✅ 完成了简单处理器的LLM对接")
        print("   ✅ 为Phase 2多模态融合奠定了基础")
        print()
        print("🚀 Ready for Phase 2: 多模态融合机制实现！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
