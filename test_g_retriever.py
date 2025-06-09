"""
G-Retriever系统测试脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag import g_retriever_system
import time

def test_g_retriever():
    """测试G-Retriever系统"""
    print("🚀 G-Retriever系统测试开始")
    print("=" * 50)
    
    # 初始化系统
    print("🔄 正在初始化G-Retriever系统...")
    if not g_retriever_system.initialize():
        print("❌ 系统初始化失败")
        return False
    
    print("✅ 系统初始化成功")
    
    # 获取系统信息
    system_info = g_retriever_system.get_system_info()
    print("\n📊 系统信息:")
    print(f"  初始化状态: {system_info['initialized']}")
    if 'stats' in system_info:
        stats = system_info['stats']
        print(f"  节点数量: {stats.get('node_count', 0)}")
        print(f"  边数量: {stats.get('edge_count', 0)}")
        print(f"  嵌入维度: {stats.get('embedding_dim', 0)}")
    
    print("\n" + "=" * 50)
    
    # 测试查询
    test_queries = [
        "勒布朗·詹姆斯的年龄是多少？",
        "科比·布莱恩特效力于哪支球队？",
        "湖人队有哪些球员？",
        "谁是最年轻的球员？",
        "告诉我关于迈克尔·乔丹的信息"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🤔 测试查询 {i}: {query}")
        print("-" * 40)
        
        start_time = time.time()
        result = g_retriever_system.retrieve_and_answer(query, format_type='qa')
        end_time = time.time()
        
        print(f"💡 答案: {result['answer']}")
        print(f"🎯 置信度: {result['confidence']:.2f}")
        print(f"⏱️  处理时间: {result['processing_time']:.2f}秒")
        
        if result['subgraph_info']:
            subgraph = result['subgraph_info']
            print(f"📊 子图规模: {subgraph['num_nodes']}个节点, {subgraph['num_edges']}条边")
        
        if result.get('seed_nodes'):
            print(f"🌱 种子节点: {result['seed_nodes']}")
        
        print()
    
    print("=" * 50)
    print("✅ G-Retriever系统测试完成")
    
    # 关闭系统
    g_retriever_system.close()
    return True

def test_individual_components():
    """测试各个组件"""
    print("🧪 测试各个组件...")
    
    # 测试图索引器
    from app.rag import GraphIndexer
    indexer = GraphIndexer()
    
    print("📊 测试图索引器...")
    if indexer.initialize():
        print("✅ 图索引器初始化成功")
        
        # 测试节点搜索
        nodes = indexer.search_nodes("勒布朗·詹姆斯", top_k=3)
        print(f"🔍 搜索结果: 找到{len(nodes)}个相关节点")
        for node in nodes:
            print(f"  - {node['name']} ({node['type']}) - 相似度: {node['similarity']:.3f}")
        
        indexer.close()
    else:
        print("❌ 图索引器初始化失败")
    
    print()

if __name__ == "__main__":
    print("🏀 篮球知识图谱G-Retriever测试")
    print("=" * 60)
    
    # 测试各个组件
    test_individual_components()
    
    print()
    
    # 测试完整系统
    try:
        test_g_retriever()
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 测试脚本执行完毕")
