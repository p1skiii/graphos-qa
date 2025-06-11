#!/usr/bin/env python3
"""
简单测试GNN组件的状态
"""
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_torch_geometric():
    """测试torch_geometric是否可用"""
    try:
        import torch_geometric
        from torch_geometric.data import Data
        logger.info(f"✅ torch_geometric 版本: {torch_geometric.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ torch_geometric 导入失败: {e}")
        return False

def test_component_factory():
    """测试组件工厂"""
    try:
        from app.rag.component_factory import component_factory
        logger.info("✅ 组件工厂导入成功")
        
        # 检查注册的组件
        graph_builders = getattr(component_factory, '_graph_builder_registry', {})
        processors = getattr(component_factory, '_processor_registry', {})
        
        logger.info(f"📊 已注册的图构建器: {list(graph_builders.keys())}")
        logger.info(f"📊 已注册的处理器: {list(processors.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 组件工厂测试失败: {e}")
        return False

def test_gnn_data_builder():
    """测试GNN数据构建器"""
    try:
        from app.rag.components.gnn_data_builder import GNNDataBuilder
        builder = GNNDataBuilder()
        logger.info("✅ GNN数据构建器导入成功")
        logger.info(f"   - 最大节点数: {builder.max_nodes}")
        logger.info(f"   - 最大跳数: {builder.max_hops}")
        logger.info(f"   - 特征维度: {builder.feature_dim}")
        return True
    except Exception as e:
        logger.error(f"❌ GNN数据构建器测试失败: {e}")
        return False

def test_gnn_processor():
    """测试GNN处理器"""
    try:
        from app.rag.processors.gnn_processor import GNNProcessor
        processor = GNNProcessor()
        logger.info("✅ GNN处理器导入成功")
        logger.info(f"   - 配置: {processor.config}")
        return True
    except Exception as e:
        logger.error(f"❌ GNN处理器测试失败: {e}")
        return False

def test_default_configs():
    """测试默认配置"""
    try:
        from app.rag.component_factory import DefaultConfigs
        
        # 测试GNN相关配置
        gnn_builder_config = DefaultConfigs.get_gnn_data_builder_config()
        gnn_processor_config = DefaultConfigs.get_gnn_processor_config()
        
        logger.info("✅ 默认配置测试成功")
        logger.info(f"   - GNN构建器配置: {gnn_builder_config.component_name}")
        logger.info(f"   - GNN处理器配置: {gnn_processor_config.component_name}")
        return True
    except Exception as e:
        logger.error(f"❌ 默认配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🧪 开始测试GNN组件...")
    
    tests = [
        ("torch_geometric", test_torch_geometric),
        ("组件工厂", test_component_factory),
        ("GNN数据构建器", test_gnn_data_builder),
        ("GNN处理器", test_gnn_processor),
        ("默认配置", test_default_configs),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n📋 测试 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
    # 总结
    logger.info("\n📊 测试结果总结:")
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"   - {test_name}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"\n🎯 总体结果: {success_count}/{total_count} 测试通过")
    
    if success_count == total_count:
        logger.info("🎉 所有GNN组件测试通过！")
        return True
    else:
        logger.warning("⚠️ 存在失败的测试，需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
