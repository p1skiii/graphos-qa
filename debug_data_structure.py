#!/usr/bin/env python3
"""
调试数据库结构，检查实际存储的球员数据
使用正确的NebulaGraph API
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.database.nebula_connection import NebulaGraphConnection

def debug_database_structure():
    """调试数据库结构"""
    print("🔍 调试NebulaGraph数据库结构...")
    
    # 初始化连接
    conn = NebulaGraphConnection()
    
    # 建立连接
    if not conn.connect():
        print("❌ 无法连接到NebulaGraph")
        return
    
    try:
        # 1. 检查所有球员（使用FETCH语法）
        print("\n📊 查询所有球员...")
        queries_to_try = [
            "SHOW TAGS",
            "FETCH PROP ON player yield vertex as v",
            "FETCH PROP ON player yield properties(vertex) as props",
            "GO FROM hash('Tim Duncan') OVER * YIELD properties($$) as props",
            "MATCH (v:player) RETURN v LIMIT 5"
        ]
        
        for i, query in enumerate(queries_to_try):
            print(f"\n尝试查询 {i+1}: {query}")
            try:
                result = conn.execute_query(query)
                
                if result and hasattr(result, 'is_succeeded') and result.is_succeeded():
                    print(f"✅ 查询成功! 行数: {result.row_size()}")
                    if hasattr(result, 'keys'):
                        print(f"列名: {result.keys()}")
                    
                    # 显示前几行结果
                    for row_idx in range(min(result.row_size(), 3)):
                        row_values = result.row_values(row_idx)
                        print(f"  行 {row_idx}: {[str(val) for val in row_values[:3]]}")  # 只显示前3列
                    
                    # 如果这个查询成功了，尝试解析数据
                    if result.row_size() > 0 and "FETCH PROP" in query:
                        print("🎯 尝试解析球员数据...")
                        for row_idx in range(min(result.row_size(), 5)):
                            row_values = result.row_values(row_idx)
                            if row_values and len(row_values) > 0:
                                try:
                                    if "properties(vertex)" in query:
                                        props = row_values[0].as_map()
                                        print(f"  球员属性: {props}")
                                    elif "vertex as v" in query:
                                        vertex = row_values[0].as_node()
                                        print(f"  球员节点: {vertex}")
                                except Exception as e:
                                    print(f"  解析失败: {e}")
                    break  # 如果成功就不再尝试其他查询
                else:
                    print("❌ 查询失败或无结果")
            except Exception as e:
                print(f"❌ 查询异常: {e}")
        
        # 2. 如果基础查询成功，尝试具体的球员查询
        print("\n🔍 搜索具体球员数据...")
        specific_queries = [
            "FETCH PROP ON player hash('Yao Ming') YIELD properties(vertex) as props",
            "FETCH PROP ON player hash('Tim Duncan') YIELD properties(vertex) as props",
            "FETCH PROP ON player hash('Tracy McGrady') YIELD properties(vertex) as props"
        ]
        
        for query in specific_queries:
            print(f"\n查询: {query}")
            try:
                result = conn.execute_query(query)
                
                if result and hasattr(result, 'is_succeeded') and result.is_succeeded():
                    if result.row_size() > 0:
                        for i in range(result.row_size()):
                            row_values = result.row_values(i)
                            if row_values and len(row_values) > 0:
                                try:
                                    props = row_values[0].as_map()
                                    print(f"  ✅ 找到球员属性: {props}")
                                except Exception as e:
                                    print(f"  解析属性失败: {e}")
                    else:
                        print("  ❌ 未找到匹配结果")
                else:
                    print("  ❌ 查询失败")
            except Exception as e:
                print(f"  ❌ 查询异常: {e}")
        
        # 3. 简化的schema检查
        print("\n🏗️ 检查数据库schema...")
        schema_queries = [
            "SHOW TAGS",
            "SHOW EDGES", 
            "DESCRIBE TAG player"
        ]
        
        for query in schema_queries:
            print(f"\n查询: {query}")
            result = conn.execute_query(query)
            
            if result and hasattr(result, 'is_succeeded') and result.is_succeeded():
                print(f"结果行数: {result.row_size()}")
                if hasattr(result, 'keys'):
                    print(f"列名: {result.keys()}")
                
                for i in range(min(result.row_size(), 5)):  # 只显示前5行
                    row_values = result.row_values(i)
                    print(f"  行 {i}: {[str(val) for val in row_values]}")
            else:
                print("  ❌ 查询失败")
        
    except Exception as e:
        print(f"❌ 数据库调试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    debug_database_structure()
