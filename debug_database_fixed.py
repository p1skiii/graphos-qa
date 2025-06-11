#!/usr/bin/env python3
"""
调试数据库查询，检查实际存储的数据结构
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.nebula_connection import NebulaGraphConnection

def debug_database():
    """调试数据库查询"""
    print("🔍 开始调试数据库查询...")
    
    # 连接数据库
    connector = NebulaGraphConnection()
    if not connector.connect():
        print("❌ 数据库连接失败")
        return
    
    print("✅ 数据库连接成功")
    
    try:
        # 1. 查看所有球员
        print("\n1. 查询所有球员:")
        query1 = "MATCH (v:player) RETURN v.player.name as name LIMIT 10"
        result1 = connector.execute_query(query1)
        
        if result1['success']:
            print(f"   找到 {result1['row_count']} 个球员:")
            for row in result1['rows']:
                name = row[0] if row[0] else "Unknown"
                print(f"   - {name}")
        else:
            print(f"   查询失败: {result1['error']}")
        
        # 2. 查看球员属性结构
        print("\n2. 查询球员属性结构:")
        query2 = "FETCH PROP ON player 'Yao Ming' YIELD properties(vertex)"
        result2 = connector.execute_query(query2)
        
        if result2['success'] and result2['row_count'] > 0:
            print("   姚明的属性:")
            row = result2['rows'][0]
            props = row[0]  # properties(vertex) returns a map
            if isinstance(props, dict):
                for key, value in props.items():
                    print(f"   - {key}: {value}")
            else:
                print(f"   属性数据: {props}")
        else:
            print(f"   没有找到姚明或查询失败: {result2.get('error', '未知错误')}")
        
        # 3. 查看所有球员的名称（更详细）
        print("\n3. 查询所有球员名称（详细）:")
        query3 = "FETCH PROP ON player * YIELD properties(vertex)"
        result3 = connector.execute_query(query3)
        
        if result3['success']:
            print(f"   找到 {result3['row_count']} 个球员:")
            for i, row in enumerate(result3['rows'][:10]):  # 限制前10个
                props = row[0]
                if isinstance(props, dict):
                    name = props.get('name', 'Unknown')
                    age = props.get('age', 'Unknown')
                    print(f"   - Name: {name}, Age: {age}")
                else:
                    print(f"   - 球员数据: {props}")
        else:
            print(f"   查询失败: {result3['error']}")
            
        # 4. 检查标签和属性
        print("\n4. 查看player标签的属性定义:")
        query4 = "DESCRIBE TAG player"
        result4 = connector.execute_query(query4)
        
        if result4['success']:
            print("   player标签属性:")
            for row in result4['rows']:
                field = row[0]
                field_type = row[1]
                print(f"   - {field}: {field_type}")
        else:
            print(f"   查询失败: {result4['error']}")
            
        # 5. 尝试搜索包含Yao的球员
        print("\n5. 搜索包含Yao的球员:")
        query5 = "MATCH (v:player) WHERE v.player.name CONTAINS 'Yao' RETURN v.player.name as name, v.player.age as age"
        result5 = connector.execute_query(query5)
        
        if result5['success']:
            print(f"   找到 {result5['row_count']} 个匹配的球员:")
            for row in result5['rows']:
                name = row[0]
                age = row[1]
                print(f"   - Name: {name}, Age: {age}")
        else:
            print(f"   查询失败: {result5['error']}")
            
    except Exception as e:
        print(f"❌ 查询过程中出现异常: {str(e)}")
        
    finally:
        connector.close()
        print("\n🔚 调试完成")

if __name__ == "__main__":
    debug_database()
