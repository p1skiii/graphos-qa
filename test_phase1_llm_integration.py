"""
Phase 1 LLM集成测试脚本
测试基础LLM框架和统一输入接口的功能
"""
import sys
import os
import time

# 添加项目路径
sys.path.append('/Users/wang/i/graphos-qa')

try:
    from app.llm import (
        LLMConfig, Phi3Config, 
        LLMEngine, create_llm_engine,
        InputRouter, UnifiedInput, LLMInputConfig, create_input_router,
        PromptTemplateManager, create_prompt_template_manager,
        ResponseFormatter, create_response_formatter,
        LLMFactory, llm_factory, create_llm_system
    )
    print("✅ 成功导入所有LLM模块")
except ImportError as e:
    print(f"❌ LLM模块导入失败: {e}")
    sys.exit(1)

def test_config_creation():
    """测试配置创建"""
    print("\n🧪 测试1: 配置创建")
    print("-" * 40)
    
    try:
        # 测试Phi-3配置
        phi3_config = Phi3Config()
        print(f"✅ Phi-3配置创建成功")
        print(f"   模型: {phi3_config.model_name}")
        print(f"   设备: {phi3_config.device}")
        print(f"   最大长度: {phi3_config.max_length}")
        
        # 测试LLM配置
        from app.llm.config import get_macos_optimized_config, validate_config
        llm_config = get_macos_optimized_config()
        print(f"✅ LLM配置创建成功")
        print(f"   多模态支持: {llm_config.enable_multimodal}")
        print(f"   最大输入tokens: {llm_config.max_input_tokens}")
        
        # 验证配置
        is_valid = validate_config(llm_config)
        print(f"✅ 配置验证: {'通过' if is_valid else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        return False

def test_input_router():
    """测试输入路由器"""
    print("\n🧪 测试2: 输入路由器")
    print("-" * 40)
    
    try:
        # 创建输入路由器
        router = create_input_router()
        print(f"✅ 输入路由器创建成功")
        
        # 测试不同类型的处理器输出
        test_cases = [
            {
                'name': 'Direct处理器输出',
                'processor_output': {
                    'textual_context': {'formatted_text': '科比是洛杉矶湖人队的传奇球员。'},
                    'metadata': {'processor_name': 'direct_processor'}
                },
                'query': '科比是谁？'
            },
            {
                'name': 'Simple G处理器输出',
                'processor_output': {
                    'textual_context': {'formatted_text': '科比和沙克是湖人队的黄金搭档。'},
                    'graph': {'nodes': [{'id': '1', 'name': '科比'}], 'edges': []},
                    'metadata': {'processor_name': 'simple_g_processor'}
                },
                'query': '科比和沙克的关系？'
            },
            {
                'name': 'Complex G处理器增强模式输出',
                'processor_output': {
                    'mode': 'enhanced',
                    'multimodal_context': None,  # 实际使用中会是MultimodalContext对象
                    'graph_embedding': {'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]},
                    'traditional_result': {
                        'textual_context': {'formatted_text': '科比和詹姆斯都是NBA巨星。'}
                    },
                    'metadata': {'processor_name': 'complex_g_processor'}
                },
                'query': '科比和詹姆斯的关系？'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n   测试案例 {i+1}: {test_case['name']}")
            unified_input = router.route_processor_output(
                test_case['processor_output'], 
                test_case['query']
            )
            print(f"   ✅ 路由成功: {unified_input.processor_type}")
            print(f"   多模态数据: {'有' if unified_input.has_multimodal_data() else '无'}")
            print(f"   文本长度: {len(unified_input.get_text_content())}")
        
        # 显示统计信息
        stats = router.get_stats()
        print(f"\n📊 路由器统计:")
        print(f"   总处理数: {stats['total_processed']}")
        print(f"   多模态比例: {stats['multimodal_ratio']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 输入路由器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_templates():
    """测试Prompt模板系统"""
    print("\n🧪 测试3: Prompt模板系统")
    print("-" * 40)
    
    try:
        # 创建模板管理器
        manager = create_prompt_template_manager()
        print(f"✅ Prompt模板管理器创建成功")
        print(f"   加载模板数: {len(manager.templates)}")
        
        # 测试模板列表
        templates = manager.list_templates()
        print(f"   可用模板: {templates}")
        
        # 测试模板格式化
        from app.llm.input_router import UnifiedInput
        
        # 创建测试输入
        test_input = UnifiedInput(
            query="科比是谁？",
            processor_type="direct",
            text_context="科比是洛杉矶湖人队的传奇球员，获得过5次总冠军。"
        )
        
        # 格式化prompt
        prompt = manager.format_prompt_for_input(test_input, "你是篮球专家")
        print(f"✅ Prompt格式化成功，长度: {len(prompt)}")
        print(f"   预览: {prompt[:150]}...")
        
        # 测试不同处理器类型
        processor_types = ['direct', 'simple_g', 'complex_g', 'comparison', 'chitchat']
        for proc_type in processor_types:
            template = manager.get_template_for_processor(proc_type)
            if template:
                print(f"   ✅ {proc_type} -> {template.name}")
            else:
                print(f"   ⚠️ {proc_type} -> 无对应模板")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt模板测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_response_formatter():
    """测试响应格式化器"""
    print("\n🧪 测试4: 响应格式化器")
    print("-" * 40)
    
    try:
        # 创建格式化器
        formatter = create_response_formatter()
        print(f"✅ 响应格式化器创建成功")
        
        # 创建测试数据
        from app.llm.llm_engine import LLMResponse
        from app.llm.input_router import UnifiedInput
        
        test_llm_response = LLMResponse(
            content="科比·布莱恩特是洛杉矶湖人队的传奇球员。他在NBA生涯中获得了5次总冠军，被誉为篮球界的传奇人物。",
            metadata={'model': 'phi-3-mini'},
            processing_time=1.5,
            token_usage={'input_tokens': 50, 'output_tokens': 100, 'total_tokens': 150}
        )
        
        test_unified_input = UnifiedInput(
            query="科比是谁？",
            processor_type="direct",
            text_context="科比相关信息"
        )
        
        # 测试格式化
        formatted_response = formatter.format_response(test_llm_response, test_unified_input)
        
        print(f"✅ 响应格式化成功")
        print(f"   原始长度: {len(test_llm_response.content)}")
        print(f"   格式化后长度: {len(formatted_response.content)}")
        print(f"   格式类型: {formatted_response.format_type}")
        print(f"   处理时间: {formatted_response.processing_info['formatting_time']:.3f}秒")
        
        # 显示格式化后的内容预览
        print(f"   内容预览: {formatted_response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 响应格式化器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_engine_creation():
    """测试LLM引擎创建（不加载模型）"""
    print("\n🧪 测试5: LLM引擎创建")
    print("-" * 40)
    
    try:
        # 创建配置
        from app.llm.config import get_macos_optimized_config
        config = get_macos_optimized_config()
        
        # 创建引擎
        engine = create_llm_engine(config)
        print(f"✅ LLM引擎创建成功")
        print(f"   模型名称: {config.model_config.model_name}")
        print(f"   设备: {config.model_config.device}")
        print(f"   多模态支持: {config.enable_multimodal}")
        print(f"   模型加载状态: {engine.is_loaded}")
        
        # 获取统计信息
        stats = engine.get_stats()
        print(f"✅ 引擎统计信息获取成功")
        print(f"   多模态启用: {stats['multimodal_enabled']}")
        print(f"   图投影器启用: {stats['graph_projector_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM引擎测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_factory():
    """测试LLM工厂"""
    print("\n🧪 测试6: LLM工厂")
    print("-" * 40)
    
    try:
        # 获取可用预设
        presets = llm_factory.list_presets()
        print(f"✅ 可用预设: {presets}")
        
        # 获取预设信息
        for preset in presets:
            info = llm_factory.get_preset_info(preset)
            if info:
                print(f"   📋 {preset}: {info['description']}")
                print(f"      模型: {info['model_name']}")
                print(f"      设备: {info['device']}")
        
        # 创建系统
        system = create_llm_system('macos_optimized')
        print(f"✅ LLM系统创建成功")
        
        # 初始化系统
        if system.initialize():
            print(f"✅ 系统初始化成功")
            
            # 获取系统状态
            status = system.get_system_status()
            print(f"   初始化状态: {status['initialized']}")
            print(f"   就绪状态: {status['ready']}")
            print(f"   组件状态: {status['components']}")
            
        else:
            print(f"❌ 系统初始化失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLM工厂测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_flow():
    """测试端到端流程（不涉及实际模型推理）"""
    print("\n🧪 测试7: 端到端流程")
    print("-" * 40)
    
    try:
        # 创建系统
        system = create_llm_system('development')
        
        if not system.initialize():
            print(f"❌ 系统初始化失败")
            return False
        
        print(f"✅ 系统初始化完成")
        
        # 模拟处理器输出
        mock_processor_output = {
            'textual_context': {
                'formatted_text': '科比·布莱恩特是NBA历史上最伟大的球员之一，他在洛杉矶湖人队效力了20年，获得了5次NBA总冠军。'
            },
            'metadata': {
                'processor_name': 'direct_processor',
                'confidence': 0.95
            },
            'success': True
        }
        
        query = "请介绍一下科比"
        
        # 测试输入路由
        unified_input = system.input_router.route_processor_output(mock_processor_output, query)
        print(f"✅ 输入路由完成: {unified_input.processor_type}")
        
        # 测试prompt生成
        prompt = system.prompt_manager.format_prompt_for_input(unified_input, "你是篮球专家")
        print(f"✅ Prompt生成完成，长度: {len(prompt)}")
        
        # 模拟LLM响应（跳过实际推理）
        from app.llm.llm_engine import LLMResponse
        mock_llm_response = LLMResponse(
            content="科比·布莱恩特（Kobe Bryant）是NBA历史上最具影响力的球员之一。他在洛杉矶湖人队度过了整个20年的职业生涯，获得了5次NBA总冠军、18次全明星选拔，并在2008年获得常规赛MVP。",
            metadata={'model': 'phi-3-mini', 'temperature': 0.7},
            processing_time=2.1,
            token_usage={'input_tokens': 120, 'output_tokens': 80, 'total_tokens': 200}
        )
        
        # 测试响应格式化
        formatted_response = system.response_formatter.format_response(mock_llm_response, unified_input)
        print(f"✅ 响应格式化完成")
        print(f"   格式类型: {formatted_response.format_type}")
        print(f"   内容长度: {len(formatted_response.content)}")
        print(f"   内容预览: {formatted_response.content[:100]}...")
        
        # 构建最终响应
        final_response = {
            'success': True,
            'content': formatted_response.content,
            'metadata': formatted_response.metadata,
            'processing_info': formatted_response.processing_info
        }
        
        print(f"✅ 端到端流程测试完成")
        print(f"   最终响应长度: {len(final_response['content'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端流程测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Phase 1 LLM集成测试")
    print("=" * 60)
    print()
    
    test_functions = [
        test_config_creation,
        test_input_router,
        test_prompt_templates,
        test_response_formatter,
        test_llm_engine_creation,
        test_llm_factory,
        test_end_to_end_flow
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 通过")
            else:
                print(f"❌ {test_func.__name__} 失败")
        except Exception as e:
            print(f"❌ {test_func.__name__} 异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！Phase 1 基础LLM集成框架实现成功！")
        print("\n📋 Phase 1 完成状态:")
        print("   ✅ LLM配置系统完成")
        print("   ✅ 统一输入接口完成")
        print("   ✅ Prompt模板系统完成")
        print("   ✅ 响应格式化器完成")
        print("   ✅ LLM引擎框架完成")
        print("   ✅ 工厂模式实现完成")
        print("   ✅ 端到端流程验证完成")
        print("\n🎯 Ready for Phase 2: 多模态融合实现！")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
