#!/usr/bin/env python3
"""
GNN组件状态检查脚本
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("🚀 开始GNN组件状态检查...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # 测试1: 基础导入
    print("🔍 测试基础导入...")
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        
        try:
            import torch_geometric
            from torch_geometric.data import Data
            print(f"✅ PyTorch Geometric 版本: {torch_geometric.__version__}")
            print("✅ torch_geometric.data.Data 导入成功")
        except ImportError:
            print("⚠️ torch_geometric 未安装，将使用简化版本")
            
        success_count += 1
    except Exception as e:
        print(f"❌ 基础导入失败: {str(e)}")
    
    print("\n" + "=" * 50)
    
    # 测试2: GNN组件
    print("🔍 测试GNN组件...")
    try:
        from app.rag.components.gnn_data_builder import GNNDataBuilder
        from app.rag.processors.gnn_processor import GNNProcessor
        print("✅ GNN组件导入成功")
        success_count += 1
    except Exception as e:
        print(f"❌ GNN组件导入失败: {str(e)}")
    
    print("\n" + "=" * 50)
    
    # 测试3: 组件工厂注册
    print("🔍 测试组件工厂注册...")
    try:
        from app.rag.components import component_factory
        print("✅ 组件工厂导入成功")
        
        # 检查注册的图构建器
        registered_builders = list(component_factory._graph_builder_registry.keys())
        print(f"📊 已注册的图构建器: {registered_builders}")
        
        if 'gnn' in registered_builders:
            print("✅ GNN数据构建器已注册")
        else:
            print("❌ GNN数据构建器未注册")
            
        # 检查处理器注册
        if hasattr(component_factory, '_processor_registry'):
            registered_processors = list(component_factory._processor_registry.keys())
            print(f"📊 已注册的处理器: {registered_processors}")
            if 'gnn' in registered_processors:
                print("✅ GNN处理器已注册")
            else:
                print("❌ GNN处理器未注册")
        else:
            print("❌ 组件工厂缺少处理器注册表")
            
        success_count += 1
    except Exception as e:
        print(f"❌ 组件工厂测试失败: {str(e)}")
    
    print("\n" + "=" * 50)
    
    # 测试4: GNN数据构建器创建
    print("🔍 测试GNN数据构建器创建...")
    try:
        from app.rag.components import DefaultConfigs
        
        # 创建GNN数据构建器配置
        config = DefaultConfigs.get_gnn_graph_builder_config()
        
        # 创建实例
        builder = component_factory.create_graph_builder(config)
        print(f"✅ 成功创建GNN数据构建器: {type(builder).__name__}")
        print(f"   - 最大节点数: {builder.max_nodes}")
        print(f"   - 最大跳数: {builder.max_hops}")
        print(f"   - 特征维度: {builder.feature_dim}")
        
        success_count += 1
    except Exception as e:
        print(f"❌ GNN数据构建器创建失败: {str(e)}")
    
    print("\n" + "=" * 50)
    
    # 总结
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")
    if success_count == total_tests:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关组件")
        return 1

if __name__ == "__main__":
    sys.exit(main())
