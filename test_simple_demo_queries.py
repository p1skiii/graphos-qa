#!/usr/bin/env python3
"""
简单Demo数据集测试
针对只有球员、球队和年龄属性的demo数据集进行英文查询测试
"""
import sys
import requests
import json
import time

def test_simple_queries():
    """测试简单的英文查询"""
    print("🏀 Testing Simple Demo Queries")
    print("="*50)
    
    # 针对demo数据集的简单英文查询
    test_queries = [
        "Who is Yao Ming?",
        "How old is Yao Ming?", 
        "What team does Yao Ming play for?",
        "Tell me about Yao Ming's age",
        "Which players are older than 30?",
        "List all teams",
        "Who plays for Lakers?",
        "What is the age of the oldest player?"
    ]
    
    base_url = "http://localhost:5000"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        try:
            # 发送查询请求
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Status: {result['status']}")
                print(f"📊 Intent: {result.get('intent', 'unknown')}")
                print(f"🔧 Processor: {result.get('processor_used', 'unknown')}")
                print(f"⏱️  Time: {result.get('processing_time', 0):.3f}s")
                
                # 显示答案
                answer = result.get('answer', 'No answer')
                llm_response = result.get('llm_response')
                
                if llm_response:
                    print(f"🤖 LLM Response: {llm_response}")
                else:
                    print(f"📝 RAG Answer: {answer}")
                
                # 显示RAG结果摘要
                rag_result = result.get('rag_result', {})
                if rag_result.get('success'):
                    context = rag_result.get('contextualized_text', '')
                    if context:
                        print(f"📋 Context: {context[:100]}...")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
        
        time.sleep(1)  # 避免请求过快

def check_system_status():
    """检查系统状态"""
    print("\n🔧 Checking System Status")
    print("="*30)
    
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ System Status: {status.get('status', 'unknown')}")
            print(f"📊 Components: {json.dumps(status.get('available_components', {}), indent=2)}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {str(e)}")
        return False
    
    return True

def initialize_pipeline():
    """初始化查询流水线"""
    print("\n⚙️ Initializing Pipeline")
    print("="*30)
    
    try:
        response = requests.post(
            "http://localhost:5000/api/pipeline/initialize",
            json={"llm_enabled": True},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Pipeline Status: {result.get('status', 'unknown')}")
            print(f"🤖 LLM Enabled: {result.get('llm_enabled', False)}")
            print(f"🔧 LLM Initialized: {result.get('llm_initialized', False)}")
            return result.get('llm_initialized', False)
        else:
            print(f"❌ Pipeline initialization failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Pipeline initialization error: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🎯 Simple Demo Dataset Query Test")
    print("For demo dataset with: Players, Teams, Age attributes only")
    print("="*60)
    
    # 1. 检查系统状态
    if not check_system_status():
        print("❌ System not available")
        return False
    
    # 2. 初始化流水线
    if not initialize_pipeline():
        print("⚠️ Pipeline initialization failed, continuing with RAG only...")
    
    # 3. 测试查询
    test_simple_queries()
    
    print("\n🎉 Test completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
