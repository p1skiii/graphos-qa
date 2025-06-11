#!/usr/bin/env python3
"""
端到端API测试脚本
使用英文查询测试完整的查询处理流水线
"""
import requests
import json
import time

# API基础URL
BASE_URL = "http://127.0.0.1:5000"

def test_health_check():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("-" * 50)

def test_component_status():
    """测试组件状态"""
    print("🔍 测试组件状态...")
    
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("-" * 50)

def test_single_retriever():
    """测试单个检索器"""
    print("🔍 测试关键词检索器...")
    
    data = {
        "query": "Kobe Bryant",
        "retriever_type": "keyword"
    }
    
    response = requests.post(f"{BASE_URL}/api/test_retriever", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("-" * 50)

def test_english_queries():
    """测试英文查询"""
    print("🔍 测试英文查询...")
    
    # 测试用例：英文查询
    test_queries = [
        "How old is Kobe Bryant?",
        "What team does LeBron James play for?",
        "Who is taller, Kobe or Jordan?",
        "Tell me about Lakers",
        "What is basketball?"
    ]
    
    for query in test_queries:
        print(f"\n🎯 查询: {query}")
        
        data = {"query": query}
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/query", json=data)
            end_time = time.time()
            
            print(f"状态码: {response.status_code}")
            print(f"处理时间: {end_time - start_time:.3f}s")
            
            result = response.json()
            
            if result.get('status') == 'success':
                print(f"✅ 成功！")
                print(f"意图: {result.get('intent')}")
                print(f"处理器: {result.get('processor_used')}")
                print(f"答案: {result.get('answer', '')[:200]}...")
                print(f"处理时间: {result.get('processing_time', 0):.3f}s")
            else:
                print(f"❌ 失败: {result.get('error')}")
            
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
        
        print("-" * 50)

def test_pipeline_stats():
    """测试流水线统计"""
    print("🔍 测试流水线统计...")
    
    response = requests.get(f"{BASE_URL}/api/pipeline/stats")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print("-" * 50)

def main():
    """主函数"""
    print("🚀 开始端到端API测试")
    print("=" * 50)
    
    try:
        # 1. 健康检查
        test_health_check()
        
        # 2. 组件状态
        test_component_status()
        
        # 3. 单个检索器测试
        test_single_retriever()
        
        # 4. 英文查询测试
        test_english_queries()
        
        # 5. 流水线统计
        test_pipeline_stats()
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器，请确保Flask应用正在运行")
    except Exception as e:
        print(f"❌ 测试异常: {str(e)}")
    
    print("🎉 测试完成")

if __name__ == "__main__":
    main()
