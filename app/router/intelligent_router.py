"""
智能路由器 - 整合篮球过滤器和BERT分类器
"""
import time
import logging
from typing import Dict, Any, List
from pathlib import Path
import json

from .basketball_filter import basketball_filter
from .intent_classifier import intent_classifier

logger = logging.getLogger(__name__)

class IntelligentRouter:
    """智能路由器"""
    
    def __init__(self):
        self.basketball_filter = basketball_filter
        self.intent_classifier = intent_classifier
        
        # 路由统计
        self.stats = {
            'total_queries': 0,
            'non_basketball_filtered': 0,
            'simple_queries': 0,
            'complex_queries': 0,
            'avg_processing_time': 0.0,
            'bert_classification_time': 0.0,
            'filter_time': 0.0
        }
        
        # 路由映射
        self.route_mapping = {
            'simple': 'direct',      # 简单查询 -> 直接检索
            'complex': 'g_retriever', # 复杂查询 -> G-Retriever
            'non_basketball': 'fallback'  # 非篮球 -> 回退处理
        }
        
        # 结果目录
        self.results_dir = Path("results/evaluations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化BERT模型
        self._initialize_bert_model()
    
    def _initialize_bert_model(self):
        """初始化BERT模型"""
        try:
            # 尝试加载微调后的模型
            model_path = "models/bert_intent_classifier_final"
            if Path(model_path).exists():
                self.intent_classifier.load_model(model_path)
                logger.info("✅ 已加载微调后的BERT模型")
            else:
                # 使用预训练模型
                self.intent_classifier.load_model()
                logger.warning("⚠️ 使用预训练BERT模型（建议先训练）")
        except Exception as e:
            logger.error(f"❌ BERT模型初始化失败: {e}")
            raise
    
    def route_query(self, user_input: str) -> Dict[str, Any]:
        """路由用户查询"""
        
        start_time = time.time()
        
        # Step 1: 篮球领域过滤
        filter_start = time.time()
        is_basketball, filter_reason, filter_analysis = self.basketball_filter.is_basketball_domain(user_input)
        filter_time = time.time() - filter_start
        
        if not is_basketball:
            # 非篮球领域，直接返回
            result = {
                'intent': 'non_basketball',
                'confidence': 0.95,
                'processor': self.route_mapping['non_basketball'],
                'reason': f'Basketball filter: {filter_reason}',
                'original_text': user_input,
                'processing_path': 'basketball_filter',
                'filter_analysis': filter_analysis,
                'filter_time': filter_time,
                'bert_time': 0.0
            }
        else:
            # Step 2: BERT意图分类
            bert_start = time.time()
            bert_intent, bert_confidence = self.intent_classifier.classify(user_input)
            bert_time = time.time() - bert_start
            
            # Step 3: 路由决策
            processor = self.route_mapping[bert_intent]
            
            result = {
                'intent': bert_intent,
                'confidence': bert_confidence,
                'processor': processor,
                'reason': f'BERT classification: {bert_intent} (conf: {bert_confidence:.3f})',
                'original_text': user_input,
                'processing_path': 'bert_classification',
                'filter_analysis': filter_analysis,
                'filter_time': filter_time,
                'bert_time': bert_time
            }
        
        # 更新统计
        total_time = time.time() - start_time
        self._update_stats(result, total_time, filter_time, result.get('bert_time', 0))
        
        # 记录日志
        self._log_routing(result, total_time)
        
        return result
    
    def batch_route(self, user_inputs: List[str]) -> List[Dict[str, Any]]:
        """批量路由"""
        
        results = []
        for user_input in user_inputs:
            result = self.route_query(user_input)
            results.append(result)
        
        return results
    
    def _update_stats(self, result: Dict, total_time: float, filter_time: float, bert_time: float):
        """更新统计信息"""
        
        self.stats['total_queries'] += 1
        
        if result['intent'] == 'non_basketball':
            self.stats['non_basketball_filtered'] += 1
        elif result['intent'] == 'simple':
            self.stats['simple_queries'] += 1
        elif result['intent'] == 'complex':
            self.stats['complex_queries'] += 1
        
        # 更新平均时间
        total = self.stats['total_queries']
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (total - 1) + total_time) / total
        )
        self.stats['filter_time'] = (
            (self.stats['filter_time'] * (total - 1) + filter_time) / total
        )
        self.stats['bert_classification_time'] = (
            (self.stats['bert_classification_time'] * (total - 1) + bert_time) / total
        )
    
    def _log_routing(self, result: Dict, total_time: float):
        """记录路由日志"""
        
        log_data = {
            'text': result['original_text'],
            'intent': result['intent'],
            'confidence': result['confidence'],
            'processor': result['processor'],
            'path': result['processing_path'],
            'time_ms': total_time * 1000,
            'filter_time_ms': result['filter_time'] * 1000,
            'bert_time_ms': result.get('bert_time', 0) * 1000,
            'reason': result['reason']
        }
        
        logger.debug(f"智能路由: {log_data}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        # 计算百分比
        enhanced_stats = {
            **self.stats,
            'non_basketball_rate': self.stats['non_basketball_filtered'] / total * 100,
            'simple_rate': self.stats['simple_queries'] / total * 100,
            'complex_rate': self.stats['complex_queries'] / total * 100,
            'basketball_filter_stats': self.basketball_filter.get_stats()
        }
        
        return enhanced_stats
    
    def get_detailed_report(self) -> str:
        """获取详细报告"""
        
        stats = self.get_stats()
        total = stats['total_queries']
        
        if total == 0:
            return "📊 暂无查询记录"
        
        report = f"""
🧠 智能路由系统报告
{'='*50}

📈 总体统计:
├── 总查询数: {total:,}
├── 平均处理时间: {stats['avg_processing_time']*1000:.1f}ms
├── 过滤器平均时间: {stats['filter_time']*1000:.1f}ms
└── BERT平均时间: {stats['bert_classification_time']*1000:.1f}ms

🎯 路由分布:
├── 非篮球查询: {stats['non_basketball_filtered']:,} ({stats['non_basketball_rate']:.1f}%) -> 回退处理
├── 简单查询: {stats['simple_queries']:,} ({stats['simple_rate']:.1f}%) -> 直接检索
└── 复杂查询: {stats['complex_queries']:,} ({stats['complex_rate']:.1f}%) -> G-Retriever

🔍 篮球过滤器详情:
├── 通过篮球过滤: {stats['basketball_filter_stats']['basketball_passed']:,}
├── 被过滤掉: {stats['basketball_filter_stats']['non_basketball_filtered']:,}
└── 过滤率: {stats['basketball_filter_stats'].get('filter_rate', 0):.1f}%

⚡ 性能指标:
├── 过滤器效率: {stats['filter_time']*1000:.1f}ms (超快速)
├── BERT分类效率: {stats['bert_classification_time']*1000:.1f}ms (可接受)
└── 总体效率: {stats['avg_processing_time']*1000:.1f}ms (良好)
        """
        
        return report
    
    def evaluate_on_test_set(self, test_cases: List[Dict]) -> Dict:
        """在测试集上评估路由准确性"""
        
        logger.info("🧪 开始路由准确性评估...")
        
        correct_routes = 0
        total_cases = len(test_cases)
        detailed_results = []
        
        for case in test_cases:
            text = case['text']
            expected_intent = case['expected_intent']
            expected_processor = case['expected_processor']
            
            # 执行路由
            result = self.route_query(text)
            
            # 检查准确性
            intent_correct = result['intent'] == expected_intent
            processor_correct = result['processor'] == expected_processor
            overall_correct = intent_correct and processor_correct
            
            if overall_correct:
                correct_routes += 1
            
            detailed_results.append({
                'text': text,
                'expected_intent': expected_intent,
                'predicted_intent': result['intent'],
                'expected_processor': expected_processor,
                'predicted_processor': result['processor'],
                'confidence': result['confidence'],
                'correct': overall_correct,
                'processing_time': result.get('filter_time', 0) + result.get('bert_time', 0)
            })
        
        # 计算准确率
        accuracy = correct_routes / total_cases if total_cases > 0 else 0
        
        evaluation_result = {
            'accuracy': accuracy,
            'correct_routes': correct_routes,
            'total_cases': total_cases,
            'detailed_results': detailed_results,
            'timestamp': time.time()
        }
        
        # 保存评估结果
        eval_path = self.results_dir / "routing_evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 路由评估结果:")
        logger.info(f"   准确率: {accuracy:.3f} ({correct_routes}/{total_cases})")
        logger.info(f"   结果已保存: {eval_path}")
        
        return evaluation_result
    
    def reset_stats(self):
        """重置统计信息"""
        
        self.stats = {
            'total_queries': 0,
            'non_basketball_filtered': 0,
            'simple_queries': 0,
            'complex_queries': 0,
            'avg_processing_time': 0.0,
            'bert_classification_time': 0.0,
            'filter_time': 0.0
        }
        
        self.basketball_filter.reset_stats()
        
        logger.info("🔄 统计信息已重置")

# 全局路由器实例
intelligent_router = IntelligentRouter()

if __name__ == "__main__":
    # 测试示例
    test_cases = [
        "姚明多少岁？",                    # Simple -> direct
        "姚明和科比什么关系？",              # Complex -> g_retriever
        "今天天气怎么样？",                  # Non-basketball -> fallback
        "科比身高是多少？",                  # Simple -> direct
        "比较湖人和凯尔特人哪个更强？",        # Complex -> g_retriever
    ]
    
    print("🧠 智能路由测试:")
    print("="*50)
    
    for query in test_cases:
        result = intelligent_router.route_query(query)
        print(f"❓ 查询: {query}")
        print(f"🎯 意图: {result['intent']}")
        print(f"📍 路由: {result['processor']}")
        print(f"💯 置信度: {result['confidence']:.3f}")
        print(f"⏱️  时间: {(result['filter_time'] + result.get('bert_time', 0))*1000:.1f}ms")
        print(f"💡 原因: {result['reason']}")
        print("-"*30)
    
    print(intelligent_router.get_detailed_report())
