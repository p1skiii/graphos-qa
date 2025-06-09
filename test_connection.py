"""
测试NebulaGraph连接的独立脚本
"""
from nebula_connection import nebula_conn

def main():
    print("🔍 开始测试NebulaGraph连接...")
    print("=" * 50)
    
    # 测试连接
    if nebula_conn.connect():
        print("✅ 连接建立成功!")
        
        # 测试基本查询
        print("\n📊 执行测试查询...")
        
        # 查询1: 显示所有标签
        print("\n1. 查询所有标签(Tags):")
        result = nebula_conn.execute_query("SHOW TAGS")
        if result['success']:
            for row in result['rows']:
                print(f"   - {row[0]}")
        else:
            print(f"   错误: {result['error']}")
        
        # 查询2: 显示所有边类型
        print("\n2. 查询所有边类型(Edges):")
        result = nebula_conn.execute_query("SHOW EDGES")
        if result['success']:
            for row in result['rows']:
                print(f"   - {row[0]}")
        else:
            print(f"   错误: {result['error']}")
        
        # 查询3: 查询部分球员数据
        print("\n3. 查询前5个球员:")
        result = nebula_conn.execute_query("MATCH (v:player) RETURN v.player.name, v.player.age LIMIT 5")
        if result['success']:
            if result['rows']:
                for row in result['rows']:
                    print(f"   - 姓名: {row[0]}, 年龄: {row[1]}")
            else:
                print("   没有找到球员数据")
        else:
            print(f"   错误: {result['error']}")
        
        # 查询4: 查询团队信息
        print("\n4. 查询前3个团队:")
        result = nebula_conn.execute_query("MATCH (v:team) RETURN v.team.name LIMIT 3")
        if result['success']:
            if result['rows']:
                for row in result['rows']:
                    print(f"   - 团队: {row[0]}")
            else:
                print("   没有找到团队数据")
        else:
            print(f"   错误: {result['error']}")
        
        print("\n🎉 所有测试完成!")
        
    else:
        print("❌ 连接失败!")
        print("\n🔧 请检查以下配置:")
        print("1. NebulaGraph是否正在运行")
        print("2. .env文件中的连接参数是否正确")
        print("3. basketballplayer space是否存在")
    
    # 清理连接
    nebula_conn.close()
    print("\n👋 测试完成，连接已关闭")

if __name__ == "__main__":
    main()
