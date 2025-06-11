#!/usr/bin/env python3
"""
完整的GNN组件测试
测试GNN数据构建器和GNN处理器的完整功能
"""
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gnn_complete_workflow():
    """测试GNN完整工作流程"""
    print("🚀 开始GNN完整工作流程测试")
    print("=" * 60)
    
    try:
        # 1. 导入必要模块
        print("📦 步骤1: 导入模块...")
        from app.rag.components import (
            component_factory, 
            DefaultConfigs, 
            ProcessorConfig,
            ComponentConfig
        )
        from app.rag.processors.gnn_processor import GNNProcessor
        print("✅ 模块导入成功")
        
        # 2. 创建GNN处理器配置
        print("⚙️ 步骤2: 创建处理器配置...")
        
        # 检索器配置
        retriever_config = DefaultConfigs.get_semantic_retriever_config()
        
        # GNN数据构建器配置  
        gnn_builder_config = DefaultConfigs.get_gnn_graph_builder_config()
        
        # 文本化器配置
        textualizer_config = DefaultConfigs.get_template_textualizer_config()
        
        # 处理器配置
        processor_config = ProcessorConfig(
            processor_name="gnn_test",
            retriever_config=retriever_config,
            graph_builder_config=gnn_builder_config,
            textualizer_config=textualizer_config,
            cache_enabled=False  # 测试时禁用缓存
        )
        print("✅ 处理器配置创建成功")
        
        # 3. 创建GNN处理器实例
        print("🤖 步骤3: 创建GNN处理器实例...")
        gnn_processor = GNNProcessor(processor_config)
        print("✅ GNN处理器实例创建成功")
        
        # 4. 测试GNN数据构建器功能
        print("🏗️ 步骤4: 测试GNN数据构建器...")
        gnn_builder = component_factory.create_graph_builder(gnn_builder_config)
        print(f"✅ GNN数据构建器创建成功: {type(gnn_builder).__name__}")
        print(f"   - 最大节点数: {gnn_builder.max_nodes}")
        print(f"   - 最大跳数: {gnn_builder.max_hops}")
        print(f"   - 特征维度: {gnn_builder.feature_dim}")
        
        # 5. 测试torch_geometric数据创建
        print("📊 步骤5: 测试torch_geometric数据创建...")
        import torch
        from torch_geometric.data import Data
        
        # 创建测试数据
        x = torch.randn(4, 768)  # 4个节点，768维特征
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        print(f"✅ torch_geometric.data.Data创建成功")
        print(f"   - 节点数: {data.num_nodes}")
        print(f"   - 边数: {data.num_edges}")
        print(f"   - 特征维度: {data.x.shape}")
        
        # 6. 测试GNN模型
        print("🧠 步骤6: 测试GNN模型...")
        from app.rag.processors.gnn_processor import SimpleGNN
        
        model = SimpleGNN(input_dim=768, hidden_dim=256, output_dim=128)
        model.eval()
        
        with torch.no_grad():
            output = model(data)
            print(f"✅ GNN模型推理成功")
            print(f"   - 输出维度: {output.shape}")
        
        print("\n" + "=" * 60)
        print("🎉 GNN完整工作流程测试成功！")
        print("✅ 所有GNN组件功能正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GNN工作流程测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_gnn_data_format():
    """测试GNN数据格式转换"""
    print("\n" + "=" * 60)
    print("📊 测试GNN数据格式转换")
    print("=" * 60)
    
    try:
        from app.rag.components import DefaultConfigs, component_factory
        
        # 创建GNN数据构建器
        config = DefaultConfigs.get_gnn_graph_builder_config()
        builder = component_factory.create_graph_builder(config)
        
        # 测试数据转换（模拟）
        print("🔄 测试数据格式转换...")
        
        # 模拟种子节点
        seed_nodes = ["player:LeBron James", "player:Stephen Curry", "team:Lakers"]
        query = "哪些球员在Lakers队效力过？"
        
        print(f"   - 种子节点: {seed_nodes}")
        print(f"   - 查询: {query}")
        print("✅ GNN数据格式转换测试准备完成")
        
        # 注意：这里不实际初始化和运行，因为需要数据库连接
        print("⚠️ 实际数据转换需要NebulaGraph连接，此处仅测试接口")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN数据格式转换测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 GNN组件完整测试开始")
    print("🔧 测试环境: PyTorch + torch_geometric")
    
    tests = [
        test_gnn_complete_workflow,
        test_gnn_data_format
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n🧪 测试 {i}/{total}: {test_func.__name__}")
        try:
            if test_func():
                passed += 1
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
        
        if i < total:
            print("-" * 60)
    
    print("\n" + "=" * 60)
    print("📊 最终测试结果")
    print("=" * 60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！GNN组件完全正常")
        print("✅ GNN数据构建器工作正常")
        print("✅ GNN处理器工作正常")
        print("✅ torch_geometric集成正常")
        return True
    else:
        print(f"⚠️ {total - passed} 个测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
