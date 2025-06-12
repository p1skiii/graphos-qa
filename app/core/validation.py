"""
QueryContext 验证器和监控工具
确保数据流的正确性和系统的可观测性
"""
from typing import List, Dict, Any, Optional
import time
import logging
from dataclasses import asdict

from .schemas import QueryContext, DataValidator

logger = logging.getLogger(__name__)

class ContextValidator:
    """上下文验证器 - 确保每个阶段的数据完整性"""
    
    def __init__(self):
        self.validation_rules = {
            "preprocessing": self._validate_preprocessing,
            "intent_classification": self._validate_intent_classification,
            "routing": self._validate_routing,
            "rag_processing": self._validate_rag_processing,
            "llm_generation": self._validate_llm_generation,
            "postprocessing": self._validate_postprocessing
        }
    
    def validate_stage(self, context: QueryContext, stage: str) -> List[str]:
        """验证特定阶段的上下文状态"""
        if stage not in self.validation_rules:
            return [f"未知的验证阶段: {stage}"]
        
        try:
            return self.validation_rules[stage](context)
        except Exception as e:
            logger.error(f"验证阶段 {stage} 时出错: {str(e)}")
            return [f"验证阶段 {stage} 失败: {str(e)}"]
    
    def validate_context(self, context: QueryContext) -> List[str]:
        """验证整个查询上下文的完整性"""
        all_errors = []
        
        # 基础验证
        if not context.request_id:
            all_errors.append("缺少请求ID")
        
        if not context.original_query or not context.original_query.strip():
            all_errors.append("原始查询为空")
        
        # 验证时间戳
        if not context.timestamp:
            all_errors.append("缺少时间戳")
        
        # 验证状态
        valid_statuses = ["processing", "success", "error"]
        if context.status not in valid_statuses:
            all_errors.append(f"无效状态: {context.status}")
        
        # 如果有错误，检查错误信息
        if context.status == "error" and not context.errors:
            all_errors.append("状态为错误但没有错误信息")
        
        # 验证各阶段数据（如果存在）
        if context.language_info and not DataValidator.validate_language_info(context.language_info):
            all_errors.append("语言信息验证失败")
        
        if context.intent_info and not DataValidator.validate_intent_info(context.intent_info):
            all_errors.append("意图信息验证失败")
        
        if context.entity_info and not DataValidator.validate_entity_info(context.entity_info):
            all_errors.append("实体信息验证失败")
        
        if context.rag_result and not DataValidator.validate_rag_result(context.rag_result):
            all_errors.append("RAG结果验证失败")
        
        if context.llm_result and not DataValidator.validate_llm_result(context.llm_result):
            all_errors.append("LLM结果验证失败")
        
        return all_errors
    
    def _validate_preprocessing(self, context: QueryContext) -> List[str]:
        """验证预处理阶段"""
        errors = []
        
        # 基础查询验证
        if not context.original_query.strip():
            errors.append("原始查询为空")
        
        # 语言信息验证
        if not context.language_info:
            errors.append("缺少语言信息")
        elif not DataValidator.validate_language_info(context.language_info):
            errors.append("语言信息格式不正确")
        
        # 标准化查询验证
        if not context.normalized_query.strip():
            errors.append("标准化查询为空")
        
        return errors
    
    def _validate_intent_classification(self, context: QueryContext) -> List[str]:
        """验证意图分类阶段"""
        errors = []
        
        # 意图信息验证
        if not context.intent_info:
            errors.append("缺少意图信息")
        elif not DataValidator.validate_intent_info(context.intent_info):
            errors.append("意图信息格式不正确")
        
        # 实体信息验证
        if not context.entity_info:
            errors.append("缺少实体信息")
        elif not context.entity_info.players and not context.entity_info.teams:
            errors.append("未提取到任何实体")
        
        return errors
    
    def _validate_routing(self, context: QueryContext) -> List[str]:
        """验证路由阶段"""
        errors = []
        
        if not context.processor_selected:
            errors.append("未选择处理器")
        
        if not context.routing_path:
            errors.append("缺少路由路径")
        
        if context.routing_time <= 0:
            errors.append("路由时间无效")
        
        return errors
    
    def _validate_rag_processing(self, context: QueryContext) -> List[str]:
        """验证RAG处理阶段"""
        errors = []
        
        if not context.rag_result:
            errors.append("缺少RAG结果")
            return errors
        
        rag = context.rag_result
        
        if not rag.success and not rag.error:
            errors.append("RAG处理失败但无错误信息")
        
        if rag.success and not rag.contextualized_text.strip():
            errors.append("RAG成功但无上下文文本")
        
        if rag.processing_time <= 0:
            errors.append("RAG处理时间无效")
        
        return errors
    
    def _validate_llm_generation(self, context: QueryContext) -> List[str]:
        """验证LLM生成阶段"""
        errors = []
        
        if not context.llm_result:
            errors.append("缺少LLM结果")
            return errors
        
        llm = context.llm_result
        
        if not llm.success and not llm.error:
            errors.append("LLM处理失败但无错误信息")
        
        if llm.success and not llm.content.strip():
            errors.append("LLM成功但无生成内容")
        
        return errors
    
    def _validate_postprocessing(self, context: QueryContext) -> List[str]:
        """验证后处理阶段"""
        errors = []
        
        if not context.final_answer.strip():
            errors.append("最终答案为空")
        
        if context.status not in ["success", "error"]:
            errors.append(f"无效的最终状态: {context.status}")
        
        if context.total_processing_time <= 0:
            errors.append("总处理时间无效")
        
        return errors
    
    def get_stats(self) -> Dict[str, Any]:
        """获取验证器统计信息"""
        return {
            "validation_rules_count": len(self.validation_rules),
            "available_stages": list(self.validation_rules.keys()),
            "last_updated": time.time()
        }

class ContextMonitor:
    """上下文监控器 - 提供系统可观测性"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0,
            "stage_times": {},
            "error_patterns": {},
            "cache_hit_rate": 0.0
        }
        self.validator = ContextValidator()
    
    def track_request(self, context: QueryContext):
        """追踪单个请求的完整生命周期"""
        self.metrics["total_requests"] += 1
        
        # 更新成功/失败计数
        if context.status == "success":
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # 更新平均处理时间
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["avg_processing_time"]
        self.metrics["avg_processing_time"] = (
            (current_avg * (total_requests - 1) + context.total_processing_time) / total_requests
        )
        
        # 追踪缓存命中率
        if context.cache_hit:
            cache_hits = sum(1 for _ in range(self.metrics["total_requests"]) if context.cache_hit)
            self.metrics["cache_hit_rate"] = cache_hits / total_requests
        
        # 记录错误模式
        for error in context.errors:
            if error not in self.metrics["error_patterns"]:
                self.metrics["error_patterns"][error] = 0
            self.metrics["error_patterns"][error] += 1
        
        # 分析处理阶段时间
        self._analyze_stage_times(context)
    
    def _analyze_stage_times(self, context: QueryContext):
        """分析各处理阶段的时间分布"""
        stage_times = {}
        
        # 从处理轨迹中提取时间信息
        if len(context.processing_trace) >= 2:
            for i in range(len(context.processing_trace) - 1):
                current = context.processing_trace[i]
                next_trace = context.processing_trace[i + 1]
                
                component = current["component"]
                time_diff = (next_trace["timestamp"] - current["timestamp"]).total_seconds()
                
                if component not in stage_times:
                    stage_times[component] = []
                stage_times[component].append(time_diff)
        
        # 更新平均时间
        for component, times in stage_times.items():
            if component not in self.metrics["stage_times"]:
                self.metrics["stage_times"][component] = {"avg": 0.0, "count": 0}
            
            current_avg = self.metrics["stage_times"][component]["avg"]
            current_count = self.metrics["stage_times"][component]["count"]
            
            for time_val in times:
                new_count = current_count + 1
                new_avg = (current_avg * current_count + time_val) / new_count
                
                self.metrics["stage_times"][component]["avg"] = new_avg
                self.metrics["stage_times"][component]["count"] = new_count
                
                current_avg = new_avg
                current_count = new_count
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        success_rate = 0.0
        if self.metrics["total_requests"] > 0:
            success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
        
        return {
            "overview": {
                "total_requests": self.metrics["total_requests"],
                "success_rate": success_rate,
                "avg_processing_time": self.metrics["avg_processing_time"],
                "cache_hit_rate": self.metrics["cache_hit_rate"]
            },
            "stage_performance": self.metrics["stage_times"],
            "error_analysis": {
                "total_errors": self.metrics["failed_requests"],
                "error_patterns": self.metrics["error_patterns"]
            }
        }
    
    def validate_and_monitor(self, context: QueryContext, stage: str) -> List[str]:
        """验证并监控特定阶段"""
        # 执行验证
        validation_errors = self.validator.validate_stage(context, stage)
        
        # 记录验证结果
        if validation_errors:
            context.add_warning("validator", f"阶段 {stage} 验证发现问题: {validation_errors}")
            logger.warning(f"Context validation failed for stage {stage}: {validation_errors}")
        else:
            context.add_trace("validator", f"stage_{stage}_validated", {"status": "passed"})
        
        # 更新监控指标
        self.track_request(context)
        
        return validation_errors

class ContextDebugger:
    """上下文调试器 - 提供详细的调试信息"""
    
    @staticmethod
    def debug_context(context: QueryContext) -> Dict[str, Any]:
        """生成详细的调试信息"""
        return {
            "basic_info": {
                "request_id": context.request_id,
                "timestamp": context.timestamp.isoformat(),
                "status": context.status,
                "total_time": context.total_processing_time
            },
            "input_analysis": {
                "original_query": context.original_query,
                "query_length": len(context.original_query),
                "language_detected": context.language_info.original_language if context.language_info else None,
                "normalized_query": context.normalized_query
            },
            "processing_stages": [
                {
                    "component": trace["component"],
                    "action": trace["action"],
                    "timestamp": trace["timestamp"].isoformat(),
                    "data": trace["data"]
                }
                for trace in context.processing_trace
            ],
            "results_summary": {
                "intent": context.intent_info.intent if context.intent_info else None,
                "entities_found": {
                    "players": len(context.entity_info.players) if context.entity_info else 0,
                    "teams": len(context.entity_info.teams) if context.entity_info else 0,
                    "attributes": len(context.entity_info.attributes) if context.entity_info else 0
                },
                "rag_success": context.rag_result.success if context.rag_result else None,
                "llm_success": context.llm_result.success if context.llm_result else None,
                "final_answer_length": len(context.final_answer)
            },
            "issues": {
                "errors": context.errors,
                "warnings": context.warnings
            }
        }
    
    @staticmethod
    def export_context_json(context: QueryContext) -> str:
        """导出上下文为JSON格式（用于日志记录）"""
        import json
        debug_data = ContextDebugger.debug_context(context)
        return json.dumps(debug_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def compare_contexts(context1: QueryContext, context2: QueryContext) -> Dict[str, Any]:
        """比较两个上下文对象的差异"""
        diff = {}
        
        # 基础比较
        if context1.original_query != context2.original_query:
            diff["query"] = {"context1": context1.original_query, "context2": context2.original_query}
        
        # 意图比较
        intent1 = context1.intent_info.intent if context1.intent_info else None
        intent2 = context2.intent_info.intent if context2.intent_info else None
        if intent1 != intent2:
            diff["intent"] = {"context1": intent1, "context2": intent2}
        
        # 处理时间比较
        if abs(context1.total_processing_time - context2.total_processing_time) > 0.1:
            diff["processing_time"] = {
                "context1": context1.total_processing_time,
                "context2": context2.total_processing_time
            }
        
        return diff

# =============================================================================
# 全局监控实例
# =============================================================================

# 创建全局监控器实例
global_monitor = ContextMonitor()
global_validator = ContextValidator()
global_debugger = ContextDebugger()
