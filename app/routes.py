"""
API路由定义 - 新组件工厂架构
"""
from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

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
                <p>请检查 templates/index.html 文件是否存在</p>
            </div>
        </body>
        </html>
        """

@api_bp.route('/api/status')
def get_status():
    """获取系统状态"""
    try:
        from app.rag.component_factory import component_factory
        
        available_components = component_factory.list_available_components()
        
        status = {
            'status': 'ok',
            'message': '新RAG组件工厂系统运行中',
            'system_type': 'component_factory_architecture',
            'available_components': available_components,
            'component_counts': {
                'retrievers': len(available_components.get('retrievers', [])),
                'graph_builders': len(available_components.get('graph_builders', [])),
                'textualizers': len(available_components.get('textualizers', []))
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'系统状态检查失败: {str(e)}'
        }), 500

@api_bp.route('/api/components')
def list_components():
    """列出所有可用组件"""
    try:
        from app.rag.component_factory import component_factory
        
        components = component_factory.list_available_components()
        
        return jsonify({
            'status': 'success',
            'components': components
        })
        
    except Exception as e:
        logger.error(f"获取组件列表失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取组件列表失败: {str(e)}'
        }), 500

@api_bp.route('/api/test_retriever', methods=['POST'])
def test_retriever():
    """测试检索器组件"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': '请提供查询内容'
            }), 400
        
        query = data['query']
        retriever_type = data.get('retriever_type', 'keyword')  # 默认使用关键词检索器
        
        # 动态创建检索器进行测试
        from app.rag.component_factory import component_factory, DefaultConfigs
        
        # 根据类型选择配置
        if retriever_type == 'semantic':
            config = DefaultConfigs.get_semantic_retriever_config()
        elif retriever_type == 'vector':
            config = DefaultConfigs.get_vector_retriever_config()
        elif retriever_type == 'keyword':
            config = DefaultConfigs.get_keyword_retriever_config()
        else:
            return jsonify({
                'status': 'error',
                'message': f'不支持的检索器类型: {retriever_type}'
            }), 400
        
        # 创建检索器
        retriever = component_factory.create_retriever(config)
        
        # 初始化检索器
        if not retriever.initialize():
            return jsonify({
                'status': 'error',
                'message': '检索器初始化失败'
            }), 500
        
        # 执行检索
        results = retriever.retrieve(query, top_k=5)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'retriever_type': retriever_type,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"检索器测试失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'检索器测试失败: {str(e)}'
        }), 500

@api_bp.route('/api/query', methods=['POST'])
def process_query():
    """处理查询请求 - 完整的端到端流水线"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': '请提供查询内容'
            }), 400
        
        query = data['query']
        context = data.get('context', None)
        
        # 导入查询流水线
        from app.api.query_pipeline import query_pipeline
        
        # 处理查询
        result = query_pipeline.process_query(query, context)
        
        # 构建响应
        if result.success:
            response = {
                'status': 'success',
                'query': result.query,
                'intent': result.intent,
                'processor_used': result.processor_used,
                'answer': result.llm_response or result.rag_result.get('contextualized_text', ''),
                'rag_result': result.rag_result,
                'llm_response': result.llm_response,
                'processing_time': result.total_time,
                'metadata': result.metadata
            }
            
            logger.info(f"✅ 查询处理成功: {query} -> {result.intent} -> {result.processor_used}")
            return jsonify(response)
        else:
            error_response = {
                'status': 'error',
                'query': result.query,
                'error': result.error,
                'processing_time': result.total_time,
                'metadata': result.metadata
            }
            
            logger.error(f"❌ 查询处理失败: {query} -> {result.error}")
            return jsonify(error_response), 500
        
    except Exception as e:
        logger.error(f"API查询处理异常: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'查询处理异常: {str(e)}'
        }), 500

@api_bp.route('/api/health')
def health_check():
    """健康检查"""
    try:
        from app.api.query_pipeline import query_pipeline
        
        health_status = query_pipeline.health_check()
        
        # 确定整体状态代码
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503
    try:
        # 检查核心组件是否可导入
        from app.rag.component_factory import component_factory
        from app.rag.cache_manager import CacheManager
        
        return jsonify({
            'status': 'healthy',
            'timestamp': 'system_ok',
            'components': {
                'component_factory': 'ok',
                'cache_manager': 'ok'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@api_bp.route('/api/pipeline/stats')
def get_pipeline_stats():
    """获取流水线统计信息"""
    try:
        from app.api.query_pipeline import query_pipeline
        
        stats = query_pipeline.get_stats()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"获取流水线统计失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取统计信息失败: {str(e)}'
        }), 500

@api_bp.route('/api/pipeline/initialize', methods=['POST'])
def initialize_pipeline():
    """初始化查询流水线"""
    try:
        data = request.get_json() or {}
        
        from app.api.query_pipeline import query_pipeline
        
        # 初始化LLM系统（如果配置启用）
        llm_config = data.get('llm_config', {})
        query_pipeline.config.update(llm_config)
        
        # 默认启用LLM
        if query_pipeline.config.get('llm_enabled', True):
            llm_success = query_pipeline.initialize_llm()
            
            return jsonify({
                'status': 'success',
                'message': '流水线初始化完成',
                'llm_enabled': query_pipeline.llm_enabled,
                'llm_initialized': llm_success
            })
        else:
            return jsonify({
                'status': 'success',
                'message': '流水线已就绪（LLM未启用）',
                'llm_enabled': False,
                'llm_initialized': False
            })
        
    except Exception as e:
        logger.error(f"流水线初始化失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'初始化失败: {str(e)}'
        }), 500

# 错误处理器
@api_bp.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'status': 'error',
        'message': 'API端点不存在',
        'available_endpoints': [
            'GET / - 首页',
            'GET /api/status - 系统状态',
            'GET /api/components - 组件列表',
            'POST /api/test_retriever - 测试检索器',
            'POST /api/query - 处理查询（开发中）',
            'GET /api/health - 健康检查'
        ]
    }), 404

@api_bp.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误',
        'note': '请检查服务器日志获取详细信息'
    }), 500
