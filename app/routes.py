"""
API路由定义
"""
from flask import Blueprint, request, jsonify
from app.rag import rag_system, g_retriever_system

# 创建蓝图
api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def home():
    """首页"""
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>篮球知识问答系统</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
                .error { color: red; padding: 20px; border: 1px solid red; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="error">
                <h2>⚠️ 模板文件缺失</h2>
                <p>请确保 templates/index.html 文件存在</p>
                <p>或者访问 /g-initialize 初始化系统</p>
            </div>
        </body>
        </html>
        """

@api_bp.route('/ask', methods=['POST'])
def ask_question():
    """处理问答请求"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        # 使用RAG系统回答问题
        result = rag_system.answer_question(question)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500

@api_bp.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_system.embedding_model is not None,
        'g_retriever_initialized': g_retriever_system.is_initialized
    })

@api_bp.route('/knowledge-base')
def get_knowledge_base():
    """获取知识库信息"""
    return jsonify({
        'total_items': len(rag_system.knowledge_base),
        'sample_items': rag_system.knowledge_base[:5] if rag_system.knowledge_base else []
    })

@api_bp.route('/g-ask', methods=['POST'])
def g_ask():
    """使用G-Retriever系统处理问答请求"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        format_type = data.get('format', 'qa')  # qa, detailed, compact
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        # 确保G-Retriever系统已初始化
        if not g_retriever_system.is_initialized:
            return jsonify({'error': 'G-Retriever系统未初始化，请先初始化系统'}), 503
        
        # 使用G-Retriever系统回答问题
        result = g_retriever_system.retrieve_and_answer(question, format_type)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500

@api_bp.route('/g-system-info')
def g_system_info():
    """获取G-Retriever系统信息"""
    try:
        info = g_retriever_system.get_system_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'获取系统信息失败: {str(e)}'}), 500

@api_bp.route('/g-initialize', methods=['POST'])
def g_initialize():
    """初始化G-Retriever系统"""
    try:
        if g_retriever_system.is_initialized:
            return jsonify({
                'status': 'already_initialized',
                'message': '系统已经初始化'
            })
        
        success = g_retriever_system.initialize()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'G-Retriever系统初始化成功'
            })
        else:
            return jsonify({
                'status': 'failed',
                'message': 'G-Retriever系统初始化失败'
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'初始化失败: {str(e)}'}), 500
