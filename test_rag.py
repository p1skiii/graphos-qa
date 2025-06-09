"""
RAG系统测试脚本
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag import rag_system

def test_rag_system():
    """测试RAG系统"""
    print("🧪 测试RAG系统...")
    print("=" * 50)
    
    # 初始化RAG系统
    if not rag_system.initialize():
        print("❌ RAG系统初始化失败")
        return
    
    # 测试问题列表
    test_questions = [
        "LeBron James多少岁？",
        "哪些球员效力于Lakers？",
        "Tim Duncan的信息",
        "有多少支球队？",
        "Kobe Bryant年龄"
    ]
    
    print(f"📊 知识库统计信息:")
    print(f"   - 总条目数: {len(rag_system.knowledge_base)}")
    print(f"   - 示例条目: {rag_system.knowledge_base[:3] if rag_system.knowledge_base else '无'}")
    print()
    
    # 测试每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"🤔 测试问题 {i}: {question}")
        print("-" * 30)
        
        result = rag_system.answer_question(question)
        
        print(f"📝 回答: {result['answer']}")
        print(f"🎯 置信度: {result['confidence']:.2f}")
        
        if result['sources']:
            print(f"📚 相关来源 (前3个):")
            for j, source in enumerate(result['sources'][:3], 1):
                print(f"   {j}. [{source['type']}] {source['text']} (相似度: {source['similarity']:.3f})")
        else:
            print("📚 无相关来源")
        
        print()
    
    # 关闭连接
    rag_system.close()
    print("✅ RAG系统测试完成!")

if __name__ == "__main__":
    test_rag_system()
