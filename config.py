"""
配置文件 - 项目全局配置
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """基础配置类"""
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 'yes']
    
    # NebulaGraph配置
    NEBULA_HOST = os.environ.get('NEBULA_HOST', '127.0.0.1')
    NEBULA_PORT = int(os.environ.get('NEBULA_PORT', 9669))
    NEBULA_USER = os.environ.get('NEBULA_USER', 'root')
    NEBULA_PASSWORD = os.environ.get('NEBULA_PASSWORD', 'nebula')
    NEBULA_SPACE = os.environ.get('NEBULA_SPACE', 'basketballplayer')
    
    # RAG配置
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    VECTOR_DB_PATH = os.environ.get('VECTOR_DB_PATH', './data/vector_db')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 512))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 50))

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
