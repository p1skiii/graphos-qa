"""
Flask应用初始化
"""
from flask import Flask
from flask_cors import CORS
from config import config

def create_app(config_name='default'):
    """应用工厂函数"""
    app = Flask(__name__)
    
    # 加载配置
    app.config.from_object(config[config_name])
    
    # 启用CORS
    CORS(app)
    
    # 注册蓝图
    from app.routes import api_bp
    app.register_blueprint(api_bp)
    
    return app
