#!/usr/bin/env python3
"""
统一数据流API测试脚本
验证新的QueryContext架构API功能
"""
import requests
import json
import time
from typing import Dict, Any

# 配置
BASE_URL = "http://localhost:5000"
API_V2_BASE = f"{BASE_URL}/api/v2"

def test_unified_api():
    """测试统一数据流API"""
    print("🚀 开始测试统一数据流API")
    print("=" * 60)
    
    # 测试1: 系统状态
    print("📊 测试1: 系统状态检查")
    try:
        response = requests.get(f"{API_V2_BASE}/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ 系统状态: {status_data['status']}")
            print(f"   架构类型: {status_data.get('system_type', 'unknown')}")
            print(f"   版本: {status_data.get('version', 'unknown')}")
            print(f"   LLM启用: {status_data.get('pipeline_status', {}).get('llm_enabled', False)}")
        else:
            print(f"❌ 状态检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 状态检查异常: {str(e)}")
    
    print()
    
    # 测试2: 健康检查
    print("🏥 测试2: 健康检查")
    try:
        response = requests.get(f"{API_V2_BASE}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 健康状态: {health_data['status']}")
            print(f"   组件状态: {health_data.get('components', {})}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 健康检查异常: {str(e)}")
    
    print()
    
    # 测试3: 单个查询
    print("🔍 测试3: 单个查询处理")
    test_queries = [
        {
            "query": "How old is Yao Ming?",
            "description": "英文年龄查询"
        },
        {
            "query": "姚明多大了？",
            "description": "中文年龄查询"
        },
        {
            "query": "Tell me about basketball",
            "description": "闲聊查询"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"  测试3.{i}: {test_case['description']}")
        try:
            payload = {
                "query": test_case["query"],
                "debug": True  # 启用调试信息
            }
            
            start_time = time.time()
            response = requests.post(f"{API_V2_BASE}/query", json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ 查询成功 ({response_time:.3f}s)")
                print(f"    请求ID: {result.get('request_id', 'unknown')}")
                print(f"    最终答案: {result.get('final_answer', 'No answer')[:100]}...")
                print(f"    处理状态: {result.get('status', 'unknown')}")
                print(f"    意图分类: {result.get('metadata', {}).get('intent', {}).get('classification', 'unknown')}")
                print(f"    使用处理器: {result.get('metadata', {}).get('routing', {}).get('processor', 'unknown')}")
                
                # 调试信息
                if result.get('debug'):
                    debug_info = result['debug']
                    print(f"    追踪步骤数: {len(debug_info.get('trace_log', []))}")
                    print(f"    错误数: {len(debug_info.get('errors', []))}")
                    print(f"    警告数: {len(debug_info.get('warnings', []))}")
            else:
                print(f"    ❌ 查询失败: {response.status_code}")
                print(f"    错误信息: {response.text}")
        except Exception as e:
            print(f"    ❌ 查询异常: {str(e)}")
        
        print()
    
    # 测试4: 批量查询
    print("📦 测试4: 批量查询处理")
    try:
        batch_payload = {
            "queries": [
                {"query": "How tall is Yao Ming?"},
                {"query": "科比多少岁？"},
                {"query": "Lakers vs Warriors"}
            ]
        }
        
        start_time = time.time()
        response = requests.post(f"{API_V2_BASE}/batch", json=batch_payload)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            print(f"✅ 批量查询完成 ({response_time:.3f}s)")
            print(f"   总查询数: {summary.get('total', 0)}")
            print(f"   成功数: {summary.get('successful', 0)}")
            print(f"   失败数: {summary.get('failed', 0)}")
            print(f"   成功率: {summary.get('success_rate', 0):.2%}")
        else:
            print(f"❌ 批量查询失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 批量查询异常: {str(e)}")
    
    print()
    
    # 测试5: 分析数据
    print("📈 测试5: 分析数据获取")
    try:
        response = requests.get(f"{API_V2_BASE}/analytics")
        if response.status_code == 200:
            analytics = response.json()
            print("✅ 分析数据获取成功")
            print(f"   性能数据: {list(analytics.get('performance', {}).keys())}")
            print(f"   验证数据: {list(analytics.get('validation', {}).keys())}")
        else:
            print(f"❌ 分析数据获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 分析数据获取异常: {str(e)}")
    
    print()
    print("🎉 统一数据流API测试完成！")

def test_error_handling():
    """测试错误处理"""
    print("⚠️ 测试错误处理")
    print("-" * 40)
    
    # 测试空查询
    print("1. 测试空查询")
    try:
        response = requests.post(f"{API_V2_BASE}/query", json={"query": ""})
        print(f"   状态码: {response.status_code}")
        if response.status_code == 400:
            print("   ✅ 正确识别空查询")
        else:
            print("   ❌ 未正确处理空查询")
    except Exception as e:
        print(f"   ❌ 异常: {str(e)}")
    
    # 测试无效JSON
    print("2. 测试无效请求")
    try:
        response = requests.post(f"{API_V2_BASE}/query", data="invalid json")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 400:
            print("   ✅ 正确处理无效请求")
        else:
            print("   ❌ 未正确处理无效请求")
    except Exception as e:
        print(f"   ❌ 异常: {str(e)}")
    
    # 测试不存在的端点
    print("3. 测试不存在的端点")
    try:
        response = requests.get(f"{API_V2_BASE}/nonexistent")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 404:
            print("   ✅ 正确处理404")
        else:
            print("   ❌ 未正确处理404")
    except Exception as e:
        print(f"   ❌ 异常: {str(e)}")

def compare_apis():
    """比较新旧API性能"""
    print("⚖️ 新旧API性能比较")
    print("-" * 40)
    
    test_query = "How old is Yao Ming?"
    
    # 测试旧API
    print("测试旧API (/api/query)")
    try:
        old_payload = {"query": test_query}
        start_time = time.time()
        old_response = requests.post(f"{BASE_URL}/api/query", json=old_payload)
        old_time = time.time() - start_time
        
        if old_response.status_code == 200:
            print(f"✅ 旧API响应时间: {old_time:.3f}s")
        else:
            print(f"❌ 旧API失败: {old_response.status_code}")
    except Exception as e:
        print(f"❌ 旧API异常: {str(e)}")
        old_time = None
    
    # 测试新API
    print("测试新API (/api/v2/query)")
    try:
        new_payload = {"query": test_query}
        start_time = time.time()
        new_response = requests.post(f"{API_V2_BASE}/query", json=new_payload)
        new_time = time.time() - start_time
        
        if new_response.status_code == 200:
            print(f"✅ 新API响应时间: {new_time:.3f}s")
        else:
            print(f"❌ 新API失败: {new_response.status_code}")
    except Exception as e:
        print(f"❌ 新API异常: {str(e)}")
        new_time = None
    
    # 性能比较
    if old_time and new_time:
        if new_time < old_time:
            improvement = ((old_time - new_time) / old_time) * 100
            print(f"🚀 新API性能提升: {improvement:.1f}%")
        else:
            degradation = ((new_time - old_time) / old_time) * 100
            print(f"⚠️ 新API性能下降: {degradation:.1f}%")

if __name__ == "__main__":
    print("🧪 统一数据流API完整测试套件")
    print("=" * 60)
    
    # 主要功能测试
    test_unified_api()
    
    print()
    
    # 错误处理测试
    test_error_handling()
    
    print()
    
    # 性能比较
    compare_apis()
    
    print()
    print("🏁 所有测试完成！")
