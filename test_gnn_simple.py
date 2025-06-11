#!/usr/bin/env python3
"""
简单的GNN组件测试脚本
测试GNN数据构建器和GNN处理器的基本功能
"""
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gnn_imports():
    """测试GNN组件导入"""
    print("=" * 60)
    print("🧪 测试GNN组件导入")
    print("=" * 60)
    
    try:
        # 测试torch_geometric导入
        import torch_geometric
        from torch_geometric.data import Data
        print(f"✅ torch_geometric导入成功，版本: {torch_geometric.__version__}")
        
        # 测试GNN数据构建器导入
        from app.rag.components.gnn_data_builder import GNNDataBuilder
        print("✅ GNN数据构建器导入成功")
        
        # 测试GNN处理器导入
        from app.rag.processors.gnn_processor import GNNProcessor
        print("✅ GNN处理器导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_component_registry():
    """测试组件注册表"""
    print("\n" + "=" * 60)
    print("🏭 测试组件注册表")
    print("=" * 60)
    
    try:
        from app.rag.component_factory import component_factory
        
        # 获取所有已注册的组件
        available_components = component_factory.list_available_components()
        
        print("📋 已注册的组件:")
        for component_type, components in available_components.items():
            print(f"  {component_type}: {components}")
        
        # 检查GNN组件是否已注册
        graph_builders = available_components.get('graph_builders', [])
        if 'gnn' in graph_builders:
            print("✅ GNN数据构建器已注册")
        else:
            print("❌ GNN数据构建器未注册")
            
        return True
        
    except Exception as e:
        print(f"❌ 组件注册表检查失败: {e}")
        return False

def test_gnn_data_builder_creation():
    """测试GNN数据构建器创建"""
    print("\n" + "=" * 60)
    print("🔧 测试GNN数据构建器创建")
    print("=" * 60)
    
    try:
        from app.rag.components.gnn_data_builder import GNNDataBuilder
        
        # 创建GNN数据构建器实例
        builder = GNNDataBuilder(
            max_nodes=10,
            max_hops=2,
            feature_dim=128
        )
        print("✅ GNN数据构建器实例创建成功")
        print(f"   - 最大节点数: {builder.max_nodes}")
        print(f"   - 最大跳数: {builder.max_hops}")
        print(f"   - 特征维度: {builder.feature_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN数据构建器创建失败: {e}")
        return False

def test_gnn_processor_creation():
    """测试GNN处理器创建"""
    print("\n" + "=" * 60)
    print("🤖 测试GNN处理器创建")
    print("=" * 60)
    
    try:
        from app.rag.processors.gnn_processor import GNNProcessor
        from app.rag.components import ProcessorDefaultConfigs
        
        # 使用默认配置创建GNN处理器
        processor = GNNProcessor()  # 使用默认配置
        print("✅ GNN处理器实例创建成功")
        print(f"   - 处理器名称: {processor.config.processor_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN处理器创建失败: {e}")
        return False

def test_torch_geometric_data():
    """测试torch_geometric.data.Data创建"""
    print("\n" + "=" * 60)
    print("📊 测试torch_geometric.data.Data创建")
    print("=" * 60)
    
    try:
        import torch
        from torch_geometric.data import Data
        
        # 创建简单的图数据
        x = torch.randn(4, 3)  # 4个节点，每个节点3维特征
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # 简单环形图
        
        data = Data(x=x, edge_index=edge_index)
        print("✅ torch_geometric.data.Data创建成功")
        print(f"   - 节点数: {data.num_nodes}")
        print(f"   - 边数: {data.num_edges}")
        print(f"   - 节点特征维度: {data.x.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ torch_geometric.data.Data创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始GNN组件简单测试")
    
    # 运行所有测试
    tests = [
        test_gnn_imports,
        test_component_registry,
        test_gnn_data_builder_creation,
        test_gnn_processor_creation,
        test_torch_geometric_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 执行失败: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！GNN组件状态良好。")
        return 0
    else:
        print("⚠️ 部分测试失败，需要检查GNN组件配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
