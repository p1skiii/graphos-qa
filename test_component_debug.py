#!/usr/bin/env python3
"""
简单的组件创建测试
"""
import sys
sys.path.append('/Users/wang/i/graphos-qa')

from app.rag.component_factory import component_factory, DefaultConfigs
from app.database.nebula_connection import nebula_conn

def test_component_creation():
    """测试组件创建"""
    print("🔍 测试组件创建...")
    
    # 确保连接已建立
    if not nebula_conn.session:
        print("📡 建立NebulaGraph连接...")
        if not nebula_conn.connect():
            print("❌ 连接失败")
            return False
        print("✅ 连接成功")
    
    try:
        # 测试创建关键词检索器
        print("\n🔧 创建关键词检索器...")
        keyword_config = DefaultConfigs.get_keyword_retriever_config()
        keyword_retriever = component_factory.create_retriever(keyword_config)
        print(f"✅ 创建成功: {keyword_retriever.__class__.__name__}")
        
        # 测试初始化
        print("🔄 初始化检索器...")
        if keyword_retriever.initialize():
            print("✅ 初始化成功")
        else:
            print("❌ 初始化失败")
            return False
        
        # 测试检索
        print("🔍 测试检索...")
        results = keyword_retriever.retrieve("Kobe Bryant", top_k=3)
        print(f"✅ 检索成功，找到 {len(results)} 个结果")
        for i, result in enumerate(results[:2]):
            print(f"  结果 {i+1}: {result.get('name', 'Unknown')} - {result.get('type', 'Unknown')}")
        
        # 测试创建简单图构建器
        print("\n🔧 创建简单图构建器...")
        simple_graph_config = DefaultConfigs.get_simple_graph_builder_config()
        simple_builder = component_factory.create_graph_builder(simple_graph_config)
        print(f"✅ 创建成功: {simple_builder.__class__.__name__}")
        
        # 测试初始化
        print("🔄 初始化图构建器...")
        if simple_builder.initialize():
            print("✅ 初始化成功")
        else:
            print("❌ 初始化失败")
            return False
        
        # 测试创建文本化器
        print("\n🔧 创建紧凑文本化器...")
        compact_config = DefaultConfigs.get_compact_textualizer_config()
        compact_textualizer = component_factory.create_textualizer(compact_config)
        print(f"✅ 创建成功: {compact_textualizer.__class__.__name__}")
        
        # 测试初始化
        print("🔄 初始化文本化器...")
        if compact_textualizer.initialize():
            print("✅ 初始化成功")
        else:
            print("❌ 初始化失败")
            return False
        
        print("\n🎉 所有组件测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 组件创建测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_creation():
    """测试处理器创建"""
    print("\n🔍 测试处理器创建...")
    
    try:
        from app.rag.processors import processor_manager
        
        # 测试创建直接处理器
        print("🔧 创建直接处理器...")
        direct_processor = processor_manager.get_processor('direct')
        print(f"✅ 处理器创建成功: {direct_processor.__class__.__name__}")
        
        # 测试查询处理
        print("🔍 测试查询处理...")
        result = direct_processor.process("How old is Kobe Bryant?")
        print(f"✅ 查询处理完成")
        print(f"成功: {result.get('success')}")
        print(f"文本: {result.get('contextualized_text', '')[:100]}...")
        
        print("\n🎉 处理器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 处理器创建测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始组件测试")
    print("=" * 50)
    
    # 测试组件创建
    if test_component_creation():
        # 测试处理器创建
        test_processor_creation()
    
    print("🎉 测试完成")

if __name__ == "__main__":
    main()
