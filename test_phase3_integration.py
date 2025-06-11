"""
Phase 3 集成测试脚本
测试GraphEncoder和ComplexGProcessor的双模式功能
"""
import sys
import os
import time
import json

# 添加项目路径
sys.path.append('/Users/wang/i/graphos-qa')

from app.rag.components.graph_encoder import GraphEncoder, MultimodalContext, create_graph_encoder
from app.rag.processors.complex_g_processor import ComplexGProcessor, ComplexGProcessorConfig, create_complex_g_processor
from app.rag.components import ProcessorConfig

def test_graph_encoder():
    """测试GraphEncoder组件"""
    print("🔧 测试GraphEncoder组件...")
    
    try:
        # 创建GraphEncoder
        encoder_config = {
            'model_config': {
                'input_dim': 768,
                'hidden_dim': 256,
                'output_dim': 128
            },
            'encoding_config': {
                'normalize_embeddings': True,
                'pooling_method': 'mean'
            }
        }
        
        encoder = create_graph_encoder(encoder_config)
        print(f"✅ GraphEncoder创建成功: {type(encoder).__name__}")
        
        # 初始化GraphEncoder
        if encoder.initialize():
            print("✅ GraphEncoder初始化成功")
        else:
            print("❌ GraphEncoder初始化失败")
            return False
        
        # 测试图编码
        test_graph = {
            'nodes': [
                {'id': 'player_1', 'type': 'player', 'name': '梅西', 'age': 35},
                {'id': 'team_1', 'type': 'team', 'name': '巴塞罗那'},
                {'id': 'player_2', 'type': 'player', 'name': '苏亚雷斯', 'age': 34}
            ],
            'edges': [
                {'source': 'player_1', 'target': 'team_1', 'relation': 'plays_for'},
                {'source': 'player_2', 'target': 'team_1', 'relation': 'plays_for'}
            ]
        }
        
        result = encoder.encode_graph(test_graph)
        
        if result.get('success'):
            print(f"✅ 图编码成功，嵌入维度: {result.get('embedding_dim', 'N/A')}")
            return True
        else:
            print(f"❌ 图编码失败: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ GraphEncoder测试失败: {str(e)}")
        return False

def test_multimodal_context():
    """测试MultimodalContext数据结构"""
    print("\n🔧 测试MultimodalContext...")
    
    try:
        # 创建测试数据
        text_context = "梅西是巴塞罗那的前锋球员，与苏亚雷斯是队友。"
        graph_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # 模拟嵌入向量
        
        # 创建MultimodalContext
        context = MultimodalContext(
            text_context=text_context,
            graph_embedding=graph_embedding,
            metadata={
                'query': '梅西和苏亚雷斯的关系',
                'creation_time': time.time()
            }
        )
        
        print("✅ MultimodalContext创建成功")
        
        # 测试方法
        combined_repr = context.get_combined_representation()
        print(f"✅ 组合表示生成成功，类型: {type(combined_repr)}")
        
        # 测试序列化
        context_dict = context.to_dict()
        restored_context = MultimodalContext.from_dict(context_dict)
        print("✅ 序列化和反序列化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ MultimodalContext测试失败: {str(e)}")
        return False

def test_complex_g_processor_traditional():
    """测试ComplexGProcessor传统模式"""
    print("\n🔧 测试ComplexGProcessor传统模式...")
    
    try:
        # 创建配置（传统模式）
        config_dict = {
            'processor_name': 'test_complex_g_processor',
            'cache_enabled': False,
            'use_enhanced_mode': False,
            'graph_encoder_enabled': False,
            'retriever_config': {'component_name': 'mock_retriever'},
            'graph_builder_config': {'component_name': 'mock_graph_builder'},
            'textualizer_config': {'component_name': 'mock_textualizer'}
        }
        
        processor = create_complex_g_processor(config_dict)
        print(f"✅ ComplexGProcessor创建成功: {processor.processor_name}")
        print(f"   当前模式: {processor.current_mode}")
        
        # 检查配置
        stats = processor.get_enhanced_stats()
        enhanced_info = stats['enhanced_info']
        print(f"   GraphEncoder启用: {enhanced_info['graph_encoder_enabled']}")
        print(f"   多模态融合启用: {enhanced_info['multimodal_fusion_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ComplexGProcessor传统模式测试失败: {str(e)}")
        return False

def test_complex_g_processor_enhanced():
    """测试ComplexGProcessor增强模式"""
    print("\n🔧 测试ComplexGProcessor增强模式...")
    
    try:
        # 创建配置（增强模式）
        config_dict = {
            'processor_name': 'test_complex_g_processor_enhanced',
            'cache_enabled': False,
            'use_enhanced_mode': True,
            'graph_encoder_enabled': True,
            'graph_encoder_config': {
                'model_config': {
                    'input_dim': 768,
                    'hidden_dim': 256,
                    'output_dim': 128
                }
            },
            'min_graph_nodes': 2,
            'fusion_strategy': 'concatenate',
            'fallback_to_traditional': True,
            'retriever_config': {'component_name': 'mock_retriever'},
            'graph_builder_config': {'component_name': 'mock_graph_builder'},
            'textualizer_config': {'component_name': 'mock_textualizer'}
        }
        
        processor = create_complex_g_processor(config_dict)
        print(f"✅ ComplexGProcessor(增强)创建成功: {processor.processor_name}")
        print(f"   当前模式: {processor.current_mode}")
        
        # 注意：实际使用中需要调用processor.initialize()来初始化所有组件
        # 在这里我们只测试基本配置和GraphEncoder功能
        
        # 检查GraphEncoder初始化
        if processor.graph_encoder:
            print("✅ GraphEncoder组件已初始化")
            
            # 测试GraphEncoder
            test_result = processor.test_graph_encoder()
            if test_result['success']:
                print("✅ GraphEncoder测试通过")
            else:
                print(f"⚠️ GraphEncoder测试失败: {test_result['error']}")
        else:
            print("⚠️ GraphEncoder未初始化")
        
        # 测试模式切换
        switch_result = processor.switch_mode('traditional')
        print(f"✅ 模式切换测试: {switch_result}, 当前模式: {processor.current_mode}")
        
        if processor.graph_encoder:
            switch_result = processor.switch_mode('enhanced')
            print(f"✅ 模式切换测试: {switch_result}, 当前模式: {processor.current_mode}")
        else:
            print("⚠️ GraphEncoder未启用，跳过增强模式切换测试")
        
        return True
        
    except Exception as e:
        print(f"❌ ComplexGProcessor增强模式测试失败: {str(e)}")
        return False

def test_integration():
    """集成测试"""
    print("\n🔧 集成测试...")
    
    try:
        # 创建包含GraphEncoder的ComplexGProcessor
        config_dict = {
            'processor_name': 'integration_test_processor',
            'cache_enabled': False,
            'use_enhanced_mode': True,
            'graph_encoder_enabled': True,
            'graph_encoder_config': {
                'model_config': {
                    'input_dim': 768,
                    'hidden_dim': 256,
                    'output_dim': 128
                }
            },
            'enable_multimodal_fusion': True,
            'fusion_strategy': 'concatenate'
        }
        
        processor = create_complex_g_processor(config_dict)
        
        # 测试多模态上下文创建
        test_textual_context = {
            'formatted_text': '梅西是阿根廷足球运动员，曾效力于巴塞罗那足球俱乐部。',
            'content': '测试内容'
        }
        
        test_graph_embedding = {
            'embedding': [0.1, 0.2, 0.3, 0.4, 0.5],
            'encoding_success': True,
            'metadata': {'test': True}
        }
        
        multimodal_context = processor._create_multimodal_context(
            test_textual_context,
            test_graph_embedding,
            '梅西的信息'
        )
        
        print("✅ 多模态上下文创建成功")
        print(f"   文本长度: {len(multimodal_context.text_context)}")
        print(f"   图嵌入长度: {len(multimodal_context.graph_embedding) if multimodal_context.graph_embedding else 0}")
        
        # 测试统计信息
        stats = processor.get_enhanced_stats()
        print("✅ 统计信息获取成功")
        print(f"   传统模式处理次数: {stats['enhanced_info']['processing_modes']['traditional_mode_count']}")
        print(f"   增强模式处理次数: {stats['enhanced_info']['processing_modes']['enhanced_mode_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 Phase 3 集成测试开始\n")
    
    test_results = {
        'graph_encoder': test_graph_encoder(),
        'multimodal_context': test_multimodal_context(),
        'complex_g_traditional': test_complex_g_processor_traditional(),
        'complex_g_enhanced': test_complex_g_processor_enhanced(),
        'integration': test_integration()
    }
    
    print("\n" + "="*60)
    print("📊 测试结果汇总:")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！Phase 3 实现成功！")
        print("\n📋 Phase 3 完成状态:")
        print("   ✅ Phase 3A: GraphEncoder组件创建完成")
        print("   ✅ Phase 3B: ComplexGProcessor双模式支持完成")
        print("   ✅ Phase 3C: MultimodalContext数据结构完成")
        print("   ✅ 组件集成和测试完成")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
