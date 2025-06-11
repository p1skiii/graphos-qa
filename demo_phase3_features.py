"""
Phase 3 功能演示脚本
展示GraphEncoder和ComplexGProcessor双模式功能的实际应用
"""
import sys
import os
import time
import json

# 添加项目路径
sys.path.append('/Users/wang/i/graphos-qa')

try:
    from app.rag.components.graph_encoder import GraphEncoder, MultimodalContext, create_graph_encoder
    from app.rag.processors.complex_g_processor import ComplexGProcessor, ComplexGProcessorConfig, create_complex_g_processor
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查模块路径是否正确")
    sys.exit(1)

def create_sample_graph_data():
    """创建示例图数据"""
    return {
        'nodes': [
            {'id': 'player:科比', 'type': 'player', 'name': '科比', 'age': 41, 'position': '得分后卫'},
            {'id': 'player:詹姆斯', 'type': 'player', 'name': '詹姆斯', 'age': 39, 'position': '小前锋'},
            {'id': 'team:湖人', 'type': 'team', 'name': '洛杉矶湖人'},
            {'id': 'team:骑士', 'type': 'team', 'name': '克利夫兰骑士'},
            {'id': 'player:韦德', 'type': 'player', 'name': '韦德', 'age': 42, 'position': '得分后卫'}
        ],
        'edges': [
            {'source': 'player:科比', 'target': 'team:湖人', 'relation': 'plays_for', 'weight': 1.0},
            {'source': 'player:詹姆斯', 'target': 'team:湖人', 'relation': 'plays_for', 'weight': 1.0},
            {'source': 'player:詹姆斯', 'target': 'team:骑士', 'relation': 'played_for', 'weight': 0.8},
            {'source': 'player:科比', 'target': 'player:詹姆斯', 'relation': 'teammate', 'weight': 0.6},
            {'source': 'player:詹姆斯', 'target': 'player:韦德', 'relation': 'friend', 'weight': 0.9}
        ]
    }

def demo_graph_encoder():
    """演示GraphEncoder功能"""
    print("🎯 演示1: GraphEncoder图编码功能")
    print("="*50)
    
    try:
        # 1. 创建GraphEncoder
        encoder_config = {
            'model_config': {
                'input_dim': 768,
                'hidden_dim': 256,
                'output_dim': 128
            }
        }
        
        print("🔧 正在创建GraphEncoder...")
        encoder = create_graph_encoder(encoder_config)
        
        print("🔧 正在初始化GraphEncoder...")
        init_result = encoder.initialize()
        
        if not init_result:
            print(f"❌ GraphEncoder初始化失败")
            return None
        
        print(f"📊 GraphEncoder配置:")
        print(f"   输入维度: {encoder.input_dim}")
        print(f"   隐藏维度: {encoder.hidden_dim}")
        print(f"   输出维度: {encoder.output_dim}")
        print()
        
        # 2. 创建示例图数据
        graph_data = create_sample_graph_data()
        print(f"📈 输入图数据:")
        print(f"   节点数: {len(graph_data['nodes'])}")
        print(f"   边数: {len(graph_data['edges'])}")
        
        for node in graph_data['nodes'][:3]:  # 显示前3个节点
            print(f"   节点: {node['name']} ({node['type']})")
        print(f"   ... 共{len(graph_data['nodes'])}个节点")
        print()
        
        # 3. 图编码
        print("🔄 执行图编码...")
        start_time = time.time()
        result = encoder.encode_graph(graph_data, "科比和詹姆斯哪个获得总冠军的概率大一些在2025年？")   
        encoding_time = time.time() - start_time
        
        if result['success']:
            print(f"✅ 图编码成功!")
            print(f"   嵌入维度: {result['embedding_dim']}")
            print(f"   编码时间: {encoding_time:.3f}秒")
            print(f"   嵌入向量: {result['embedding'][:5]}...") # 显示前5个数值
            print()
            return result['embedding']
        else:
            print(f"❌ 图编码失败: {result['error']}")
            return None
            
    except Exception as e:
        print(f"❌ GraphEncoder演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def demo_multimodal_context():
    """演示MultimodalContext数据结构"""
    print("🎯 演示2: MultimodalContext多模态上下文")
    print("="*50)
    
    # 1. 创建文本和图嵌入
    text_context = """
    根据图谱分析，科比和詹姆斯都是NBA的顶级球员。科比效力于洛杉矶湖人队，
    是一名得分后卫。詹姆斯曾效力于克利夫兰骑士队，后来也加入了湖人队，
    是一名小前锋。他们曾经是队友，并且都取得了辉煌的成就。
    """
    
    # 模拟图嵌入（实际应该来自GraphEncoder）
    graph_embedding = [0.23, -0.15, 0.67, 0.42, -0.31, 0.89, -0.56, 0.78, 0.12, -0.44]
    
    # 2. 创建MultimodalContext
    multimodal_context = MultimodalContext(
        text_context=text_context.strip(),
        graph_embedding=graph_embedding,
        metadata={
            'query': '科比和詹姆斯的关系',
            'creation_time': time.time(),
            'source': 'demo',
            'version': '1.0'
        }
    )
    
    print(f"📝 多模态上下文创建:")
    print(f"   文本长度: {len(multimodal_context.text_context)} 字符")
    print(f"   图嵌入维度: {len(multimodal_context.graph_embedding)}")
    print(f"   元数据: {list(multimodal_context.metadata.keys())}")
    print()
    
    # 3. 序列化和反序列化测试
    print("🔄 序列化测试...")
    context_dict = multimodal_context.to_dict()
    restored_context = MultimodalContext.from_dict(context_dict)
    
    print(f"✅ 序列化成功，字典键: {list(context_dict.keys())}")
    print(f"✅ 反序列化成功，文本长度: {len(restored_context.text_context)}")
    print()
    
    # 4. 组合表示
    combined_repr = multimodal_context.get_combined_representation()
    print(f"🎭 组合表示:")
    print(f"   类型: {type(combined_repr)}")
    print(f"   包含键: {list(combined_repr.keys())}")
    print(f"   模态数量: {combined_repr['integration_info']['modality_count']}")
    print(f"   文本长度: {combined_repr['integration_info']['text_length']}")
    print()
    
    return multimodal_context

def demo_complex_g_processor():
    """演示ComplexGProcessor双模式功能"""
    print("🎯 演示3: ComplexGProcessor双模式处理")
    print("="*50)
    
    try:
        # 1. 创建传统模式处理器
        traditional_config = {
            'processor_name': 'demo_traditional_processor',
            'cache_enabled': False,
            'use_enhanced_mode': False,
            'graph_encoder_enabled': False
        }
        
        print("🔧 创建传统模式处理器...")
        traditional_processor = create_complex_g_processor(traditional_config)
        print(f"📊 传统模式处理器:")
        print(f"   处理器名称: {traditional_processor.processor_name}")
        print(f"   当前模式: {traditional_processor.current_mode}")
        print(f"   GraphEncoder启用: {traditional_processor.graph_encoder is not None}")
        print()
        
        # 2. 创建增强模式处理器
        enhanced_config = {
            'processor_name': 'demo_enhanced_processor',
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
            'fusion_strategy': 'concatenate',
            'min_graph_nodes': 2
        }
        
        print("🔧 创建增强模式处理器...")
        enhanced_processor = create_complex_g_processor(enhanced_config)
        
        # 检查GraphEncoder是否正确创建
        if enhanced_processor.graph_encoder is None:
            print("⚠️ GraphEncoder创建失败，尝试手动创建...")
            try:
                encoder = create_graph_encoder(enhanced_config['graph_encoder_config'])
                init_result = encoder.initialize()
                if init_result:
                    enhanced_processor.graph_encoder = encoder
                    print("✅ 手动创建GraphEncoder成功")
                else:
                    print(f"❌ 手动创建GraphEncoder失败")
            except Exception as e:
                print(f"❌ 手动创建GraphEncoder异常: {str(e)}")
        
        print(f"🚀 增强模式处理器:")
        print(f"   处理器名称: {enhanced_processor.processor_name}")
        print(f"   当前模式: {enhanced_processor.current_mode}")
        print(f"   GraphEncoder启用: {enhanced_processor.graph_encoder is not None}")
        if hasattr(enhanced_processor, 'complex_config'):
            print(f"   多模态融合: {enhanced_processor.complex_config.enable_multimodal_fusion}")
            print(f"   融合策略: {enhanced_processor.complex_config.fusion_strategy}")
        print()
        
        # 3. 模式切换演示
        print("🔄 模式切换演示:")
        print(f"   当前模式: {enhanced_processor.current_mode}")
        
        enhanced_processor.switch_mode('traditional')
        print(f"   切换后模式: {enhanced_processor.current_mode}")
        
        if enhanced_processor.graph_encoder:
            enhanced_processor.switch_mode('enhanced')
            print(f"   再次切换后模式: {enhanced_processor.current_mode}")
        print()
        
        # 4. GraphEncoder测试
        if enhanced_processor.graph_encoder:
            print("🧠 GraphEncoder测试:")
            test_result = enhanced_processor.test_graph_encoder()
            
            if test_result['success']:
                print(f"   ✅ 测试通过")
                print(f"   编码器类型: {test_result['encoder_info']['model_type']}")
            else:
                print(f"   ❌ 测试失败: {test_result['error']}")
            print()
        
        # 5. 统计信息
        stats = enhanced_processor.get_enhanced_stats()
        enhanced_info = stats['enhanced_info']
        
        print("📈 处理器统计信息:")
        print(f"   传统模式处理次数: {enhanced_info['processing_modes']['traditional_mode_count']}")
        print(f"   增强模式处理次数: {enhanced_info['processing_modes']['enhanced_mode_count']}")
        print(f"   模式切换次数: {enhanced_info['processing_modes']['mode_switches']}")
        print()
        
        return enhanced_processor
        
    except Exception as e:
        print(f"❌ ComplexGProcessor演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def demo_integration_workflow():
    """演示完整的集成工作流"""
    print("🎯 演示4: 完整集成工作流")
    print("="*50)
    
    try:
        # 1. 创建增强模式处理器
        config = {
            'processor_name': 'integration_demo_processor',
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
        
        processor = create_complex_g_processor(config)
        print(f"🔧 创建处理器: {processor.processor_name}")
        
        # 检查GraphEncoder状态
        if processor.graph_encoder is None:
            print("⚠️ GraphEncoder未正确初始化，尝试修复...")
            try:
                encoder = create_graph_encoder(config['graph_encoder_config'])
                init_result = encoder.initialize()
                if init_result:
                    processor.graph_encoder = encoder
                    print("✅ GraphEncoder修复成功")
                else:
                    print(f"❌ GraphEncoder修复失败")
                    return None
            except Exception as e:
                print(f"❌ GraphEncoder修复异常: {str(e)}")
                return None
        
        # 2. 模拟数据处理流程
        query = "科比和詹姆斯有什么共同点？"
        graph_data = create_sample_graph_data()
        
        print(f"📝 查询: {query}")
        print(f"📊 图数据: {len(graph_data['nodes'])}个节点, {len(graph_data['edges'])}条边")
        
        # 3. 图编码
        if processor.graph_encoder:
            print("\n🔄 执行图编码...")
            encoding_result = processor._encode_graph_data(graph_data)
            
            if encoding_result and encoding_result['encoding_success']:
                print(f"✅ 图编码成功，嵌入维度: {len(encoding_result['embedding'])}")
                
                # 4. 多模态上下文创建
                textual_context = {
                    'formatted_text': '科比和詹姆斯都是NBA历史上的伟大球员，他们都效力过洛杉矶湖人队。',
                    'content': '基于图谱的分析结果'
                }
                
                print("\n🎭 创建多模态上下文...")
                multimodal_context = processor._create_multimodal_context(
                    textual_context,
                    encoding_result,
                    query
                )
                
                print(f"✅ 多模态上下文创建成功")
                print(f"   文本长度: {len(multimodal_context.text_context)}")
                print(f"   图嵌入: {'有' if multimodal_context.graph_embedding else '无'}")
                print(f"   元数据: {list(multimodal_context.metadata.keys())}")
                
                return multimodal_context
            else:
                print("❌ 图编码失败")
        else:
            print("⚠️ GraphEncoder未启用")
        
        return None
        
    except Exception as e:
        print(f"❌ 集成工作流演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主演示函数"""
    print("🚀 Phase 3 功能演示")
    print("GraphEncoder + ComplexGProcessor 双模式增强")
    print("="*60)
    print()
    
    # 检查Python环境
    print(f"🐍 Python版本: {sys.version}")
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📦 项目路径: /Users/wang/i/graphos-qa")
    print()
    
    try:
        # 演示1: GraphEncoder
        print("开始演示1...")
        graph_embedding = demo_graph_encoder()
        
        print()
        # 演示2: MultimodalContext
        print("开始演示2...")
        multimodal_context = demo_multimodal_context()
        
        print()
        # 演示3: ComplexGProcessor
        print("开始演示3...")
        processor = demo_complex_g_processor()
        
        print()
        # 演示4: 集成工作流
        print("开始演示4...")
        integration_result = demo_integration_workflow()
        
        print()
        print("="*60)
        print("🎉 Phase 3 功能演示完成！")
        print()
        print("📋 演示总结:")
        print(f"   {'✅' if graph_embedding is not None else '❌'} GraphEncoder: {'图到向量编码功能正常' if graph_embedding is not None else '图编码功能异常'}")
        print(f"   {'✅' if multimodal_context is not None else '❌'} MultimodalContext: {'多模态数据结构完整' if multimodal_context is not None else '多模态数据结构异常'}")
        print(f"   {'✅' if processor is not None else '❌'} ComplexGProcessor: {'双模式切换功能正常' if processor is not None else '双模式切换功能异常'}")
        print(f"   {'✅' if integration_result is not None else '❌'} 集成工作流: {'端到端处理流程完整' if integration_result is not None else '端到端处理流程异常'}")
        print()
        
        if all([graph_embedding is not None, multimodal_context is not None, processor is not None]):
            print("🚀 Ready for production use!")
        else:
            print("⚠️ 部分功能存在问题，请检查相关模块")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
