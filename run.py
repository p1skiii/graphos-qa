"""
项目启动入口文件
"""
import os
from app import create_app
from app.rag import rag_system

def main():
    """主函数"""
    print("🚀 正在启动篮球知识问答系统...")
    
    # 创建Flask应用
    app = create_app('development')
    
    # 初始化RAG系统
    if not rag_system.initialize():
        print("❌ RAG系统初始化失败，程序退出")
        return
    
    print("✅ 系统启动成功!")
    print("🌐 访问地址: http://127.0.0.1:5000")
    print("💡 您可以问一些关于篮球的问题，比如：")
    print("   - LeBron James多少岁？")
    print("   - 哪些球员效力于Lakers？")
    print("   - Tim Duncan的信息")
    
    try:
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG']
        )
    except KeyboardInterrupt:
        print("\n👋 正在关闭系统...")
    finally:
        # 清理资源
        rag_system.close()
        print("✅ 系统已安全关闭")

if __name__ == '__main__':
    main()
