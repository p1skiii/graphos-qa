#!/usr/bin/env python3
"""
Simple English Demo Test for Basketball Knowledge QA System
Test with demo dataset (only players and teams, player age attribute only)
"""
import requests
import json
import time
import sys
import os

# Add project path
sys.path.append('/Users/wang/i/graphos-qa')

def test_api_endpoint():
    """Test the main API endpoint with simple English queries"""
    base_url = "http://127.0.0.1:5000"
    
    # Simple English queries for demo dataset
    test_queries = [
        {
            "query": "How old is Yao Ming?",
            "expected_intent": "direct_db_lookup",
            "description": "Simple age query about Yao Ming"
        },
        {
            "query": "What is the age of Yao Ming?", 
            "expected_intent": "direct_db_lookup",
            "description": "Age query variation"
        },
        {
            "query": "Tell me about Lakers",
            "expected_intent": "direct_db_lookup", 
            "description": "Team information query"
        },
        {
            "query": "Who is Yao Ming?",
            "expected_intent": "direct_db_lookup",
            "description": "Player information query"
        },
        {
            "query": "What teams are there?",
            "expected_intent": "direct_db_lookup",
            "description": "Teams listing query"
        }
    ]
    
    print("🏀 Testing Basketball Knowledge QA System - Simple English Demo")
    print("=" * 70)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Server health check: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   Components: {health_data.get('components', {})}")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Please start the Flask server first:")
        print("  python run.py")
        return False
    
    print("\n🔍 Running Query Tests")
    print("-" * 50)
    
    success_count = 0
    total_queries = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_intent = test_case["expected_intent"]
        description = test_case["description"]
        
        print(f"\n{i}. {description}")
        print(f"   Query: \"{query}\"")
        
        try:
            # Send query
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    print(f"   ✅ Status: Success")
                    print(f"   🎯 Intent: {data.get('intent')}")
                    print(f"   📋 Processor: {data.get('processor_used')}")
                    print(f"   ⏱️  Time: {data.get('processing_time', 0):.3f}s")
                    
                    # Check if we have an answer
                    answer = data.get('answer', '')
                    llm_response = data.get('llm_response')
                    
                    if llm_response:
                        print(f"   🤖 LLM Response: {llm_response[:100]}...")
                        success_count += 1
                    elif answer:
                        print(f"   📄 RAG Answer: {answer[:100]}...")
                        success_count += 1
                    else:
                        print(f"   ⚠️  No answer provided")
                        
                    # Show RAG result if available
                    rag_result = data.get('rag_result', {})
                    if rag_result.get('contextualized_text'):
                        context = rag_result['contextualized_text'][:100]
                        print(f"   🔍 RAG Context: {context}...")
                        
                else:
                    print(f"   ❌ Status: Error - {data.get('error', 'Unknown error')}")
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                if response.text:
                    print(f"      Response: {response.text[:200]}")
                    
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
        
        time.sleep(0.5)  # Small delay between requests
    
    print("\n" + "=" * 70)
    print(f"📊 Test Summary: {success_count}/{total_queries} queries successful")
    print(f"   Success Rate: {success_count/total_queries*100:.1f}%")
    
    return success_count == total_queries

def check_system_stats():
    """Check system statistics"""
    base_url = "http://127.0.0.1:5000"
    
    try:
        response = requests.get(f"{base_url}/api/pipeline/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("\n📊 System Statistics:")
            print("-" * 30)
            
            pipeline_stats = stats.get('pipeline_stats', {})
            print(f"Total Queries: {pipeline_stats.get('total_queries', 0)}")
            print(f"Successful: {pipeline_stats.get('successful_queries', 0)}")
            print(f"Failed: {pipeline_stats.get('failed_queries', 0)}")
            print(f"Avg Time: {pipeline_stats.get('avg_processing_time', 0):.3f}s")
            
            llm_enabled = stats.get('llm_enabled', False)
            print(f"LLM Enabled: {llm_enabled}")
            
        else:
            print(f"❌ Could not get stats: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Stats check failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Simple English Demo Test")
    
    # Run the main test
    success = test_api_endpoint()
    
    # Check system stats
    check_system_stats()
    
    if success:
        print("\n🎉 All tests passed! The system is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1)
