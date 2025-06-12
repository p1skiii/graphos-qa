"""
统一数据流API路由
基于QueryContext架构的新一代API接口
"""
from flask import Blueprint, request, jsonify
import logging
import time
from typing import Dict, Any

from app.api.unified_pipeline import UnifiedQueryPipeline
from app.core.schemas import QueryContext, QueryContextFactory
from app.core.validation import global_monitor, global_validator

logger = logging.getLogger(__name__)

# 创建蓝图
unified_api_bp = Blueprint('unified_api', __name__)

# 全局流水线实例
_pipeline = None

def get_pipeline() -> UnifiedQueryPipeline:
    """获取或创建流水线实例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedQueryPipeline()
    return _pipeline

@unified_api_bp.route('/api/v2/status')
def get_unified_status():
    """获取统一系统状态"""
    try:
        pipeline = get_pipeline()
        
        # 获取监控信息
        monitor_report = global_monitor.get_performance_report()
        
        status = {
            'status': 'ok',
            'message': '统一数据流架构运行中',
            'system_type': 'unified_query_context_architecture',
            'version': '2.0',
            'architecture': {
                'data_backbone': 'QueryContext',
                'validation': 'enabled',
                'monitoring': 'enabled',
                'tracing': 'enabled'
            },
            'pipeline_status': {
                'llm_enabled': pipeline.llm_enabled,
                'llm_ready': pipeline.llm_system.is_ready if pipeline.llm_system else False,
                'router_initialized': pipeline.router is not None,
                'processor_mapping': pipeline.processor_mapping
            },
            'performance': monitor_report,
            'timestamp': time.time()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ 获取系统状态失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'系统状态检查失败: {str(e)}',
            'timestamp': time.time()
        }), 500

@unified_api_bp.route('/api/v2/query', methods=['POST'])
def process_unified_query():
    """统一查询处理接口"""
    try:
        # 解析请求
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({
                'error': '无效的JSON格式',
                'status': 'error',
                'details': str(e)
            }), 400
        
        if not data:
            return jsonify({
                'error': '请求数据为空',
                'status': 'error'
            }), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({
                'error': '查询内容不能为空',
                'status': 'error'
            }), 400
        
        # 获取用户上下文
        user_context = data.get('context', {})
        
        # 处理查询
        pipeline = get_pipeline()
        context = pipeline.process_query_unified(query, user_context)
        
        # 验证结果
        validation_errors = global_validator.validate_context(context)
        if validation_errors:
            logger.warning(f"⚠️ 上下文验证发现问题: {validation_errors}")
        
        # 构建响应
        response = {
            'request_id': context.request_id,
            'query': context.original_query,
            'final_answer': context.final_answer,
            'status': context.status,
            'processing_time': context.total_processing_time,
            'metadata': {
                'language': {
                    'original': context.language_info.original_language if context.language_info else 'unknown',
                    'answer': context.answer_language
                },
                'intent': {
                    'classification': context.intent_info.intent if context.intent_info else 'unknown',
                    'confidence': context.intent_info.confidence if context.intent_info else 0.0
                },
                'routing': {
                    'path': getattr(context, 'routing_path', 'unknown'),
                    'processor': getattr(context, 'processor_selected', 'unknown'),
                    'reason': getattr(context, 'routing_reason', 'unknown')
                },
                'rag': {
                    'success': context.rag_result.success if context.rag_result else False,
                    'processor_used': context.rag_result.processor_used if context.rag_result else 'unknown',
                    'nodes_retrieved': len(context.rag_result.retrieved_nodes) if context.rag_result and context.rag_result.retrieved_nodes else 0,
                    'confidence': context.rag_result.confidence if context.rag_result else 0.0
                },
                'llm': {
                    'success': context.llm_result.success if context.llm_result else False,
                    'model_used': context.llm_result.model_used if context.llm_result else 'unknown',
                    'tokens_used': context.llm_result.tokens_used if context.llm_result else 0
                },
                'tracing': {
                    'total_steps': len(context.trace_log),
                    'errors': len(context.errors),
                    'warnings': len(context.warnings)
                }
            }
        }
        
        # 添加调试信息（可选）
        if data.get('debug', False):
            response['debug'] = {
                'trace_log': context.trace_log,
                'errors': context.errors,
                'warnings': context.warnings,
                'validation_errors': validation_errors
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ 查询处理失败: {str(e)}")
        return jsonify({
            'error': f'查询处理失败: {str(e)}',
            'status': 'error',
            'timestamp': time.time()
        }), 500

@unified_api_bp.route('/api/v2/batch', methods=['POST'])
def process_batch_queries():
    """批量查询处理接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': '请求数据为空',
                'status': 'error'
            }), 400
        
        queries = data.get('queries', [])
        if not queries:
            return jsonify({
                'error': '查询列表不能为空',
                'status': 'error'
            }), 400
        
        if len(queries) > 10:  # 限制批量查询数量
            return jsonify({
                'error': '批量查询数量不能超过10个',
                'status': 'error'
            }), 400
        
        # 处理批量查询
        pipeline = get_pipeline()
        results = []
        
        for i, query_data in enumerate(queries):
            try:
                query = query_data.get('query', '').strip() if isinstance(query_data, dict) else str(query_data).strip()
                user_context = query_data.get('context', {}) if isinstance(query_data, dict) else {}
                
                if not query:
                    results.append({
                        'index': i,
                        'error': '查询内容不能为空',
                        'status': 'error'
                    })
                    continue
                
                # 处理单个查询
                context = pipeline.process_query_unified(query, user_context)
                
                results.append({
                    'index': i,
                    'request_id': context.request_id,
                    'query': context.original_query,
                    'final_answer': context.final_answer,
                    'status': context.status,
                    'processing_time': context.total_processing_time
                })
                
            except Exception as e:
                logger.error(f"❌ 批量查询第{i}项处理失败: {str(e)}")
                results.append({
                    'index': i,
                    'error': f'处理失败: {str(e)}',
                    'status': 'error'
                })
        
        # 统计结果
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0
            },
            'status': 'completed',
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ 批量查询处理失败: {str(e)}")
        return jsonify({
            'error': f'批量查询处理失败: {str(e)}',
            'status': 'error',
            'timestamp': time.time()
        }), 500

@unified_api_bp.route('/api/v2/analytics', methods=['GET'])
def get_analytics():
    """获取系统分析数据"""
    try:
        # 获取监控报告
        monitor_report = global_monitor.get_performance_report()
        
        # 获取验证器统计
        validator_stats = global_validator.get_stats()
        
        analytics = {
            'performance': monitor_report,
            'validation': validator_stats,
            'timestamp': time.time()
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"❌ 获取分析数据失败: {str(e)}")
        return jsonify({
            'error': f'获取分析数据失败: {str(e)}',
            'status': 'error',
            'timestamp': time.time()
        }), 500

@unified_api_bp.route('/api/v2/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        pipeline = get_pipeline()
        
        health_status = {
            'status': 'healthy',
            'components': {
                'pipeline': 'ok',
                'router': 'ok' if pipeline.router else 'error',
                'llm_system': 'ok' if pipeline.llm_system and pipeline.llm_system.is_ready else 'warning',
                'monitor': 'ok',
                'validator': 'ok'
            },
            'timestamp': time.time()
        }
        
        # 检查是否有问题
        has_errors = any(status == 'error' for status in health_status['components'].values())
        has_warnings = any(status == 'warning' for status in health_status['components'].values())
        
        if has_errors:
            health_status['status'] = 'unhealthy'
            return jsonify(health_status), 503
        elif has_warnings:
            health_status['status'] = 'degraded'
            return jsonify(health_status), 200
        else:
            return jsonify(health_status), 200
            
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@unified_api_bp.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'error': '接口不存在',
        'status': 'error',
        'available_endpoints': [
            '/api/v2/status',
            '/api/v2/query',
            '/api/v2/batch',
            '/api/v2/analytics',
            '/api/v2/health'
        ],
        'timestamp': time.time()
    }), 404

@unified_api_bp.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"❌ 内部服务器错误: {str(error)}")
    return jsonify({
        'error': '内部服务器错误',
        'status': 'error',
        'timestamp': time.time()
    }), 500
