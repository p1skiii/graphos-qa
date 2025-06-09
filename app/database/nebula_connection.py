"""
NebulaGraph连接管理器
"""
import os
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from config import Config as AppConfig

class NebulaGraphConnection:
    def __init__(self):
        self.connection_pool = None
        self.session = None
        
    def connect(self):
        """建立与NebulaGraph的连接"""
        try:
            # 从配置获取连接参数
            host = AppConfig.NEBULA_HOST
            port = AppConfig.NEBULA_PORT
            user = AppConfig.NEBULA_USER
            password = AppConfig.NEBULA_PASSWORD
            space = AppConfig.NEBULA_SPACE
            
            # 创建连接配置
            config = Config()
            config.max_connection_pool_size = 10
            
            # 创建连接池
            self.connection_pool = ConnectionPool()
            ok = self.connection_pool.init([(host, port)], config)
            
            if not ok:
                raise Exception("连接池初始化失败")
            
            # 获取会话
            self.session = self.connection_pool.get_session(user, password)
            
            # 切换到指定space
            resp = self.session.execute(f'USE {space}')
            if not resp.is_succeeded():
                raise Exception(f"切换到space {space} 失败: {resp.error_msg()}")
                
            print(f"成功连接到NebulaGraph space: {space}")
            return True
            
        except Exception as e:
            print(f"连接NebulaGraph失败: {str(e)}")
            return False
    
    def execute_query(self, query):
        """执行nGQL查询"""
        if not self.session:
            raise Exception("未建立连接，请先调用connect()方法")
        
        try:
            resp = self.session.execute(query)
            if resp.is_succeeded():
                # 使用正确的NebulaGraph API
                column_names = resp.keys()  # 获取列名
                row_count = resp.row_size()  # 获取行数
                rows = []
                
                # 遍历每一行数据
                for i in range(row_count):
                    row_values = resp.row_values(i)  # 获取第i行的值列表
                    row_data = []
                    
                    for value in row_values:
                        # 根据值的类型进行处理
                        if value.is_string():
                            row_data.append(value.as_string())
                        elif value.is_int():
                            row_data.append(value.as_int())
                        elif value.is_double():
                            row_data.append(value.as_double())
                        elif value.is_bool():
                            row_data.append(value.as_bool())
                        elif value.is_map():
                            row_data.append(value.as_map())
                        elif value.is_node():
                            node = value.as_node()
                            row_data.append(node)
                        elif value.is_relationship():
                            rel = value.as_relationship()
                            row_data.append(rel)
                        else:
                            # 其他类型转为字符串
                            row_data.append(value.as_string())
                    
                    rows.append(row_data)
                
                return {
                    'success': True,
                    'data': resp,
                    'column_names': column_names,
                    'rows': rows,
                    'row_count': row_count
                }
            else:
                return {
                    'success': False,
                    'error': resp.error_msg()
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self):
        """关闭连接"""
        if self.session:
            self.session.release()
        if self.connection_pool:
            self.connection_pool.close()
        print("NebulaGraph连接已关闭")
    
    def test_connection(self):
        """测试连接"""
        try:
            # 执行简单查询测试连接
            result = self.execute_query("SHOW TAGS")
            if result['success']:
                print("连接测试成功！")
                print("可用的标签(Tags):")
                for row in result['rows']:
                    print(f"  - {row[0]}")
                return True
            else:
                print(f"连接测试失败: {result['error']}")
                return False
        except Exception as e:
            print(f"连接测试异常: {str(e)}")
            return False

# 全局连接实例
nebula_conn = NebulaGraphConnection()
