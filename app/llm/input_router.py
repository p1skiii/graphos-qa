"""
LLM输入路由器
实现统一输入接口，支持来自不同RAG处理器的输出
基于G-Retriever的多模态融合方法
"""
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.rag.components.graph_encoder import MultimodalContext

logger = logging.getLogger(__name__)

@dataclass
class LLMInputConfig:
    """LLM输入配置"""
    processor_type: str  # direct, simple_g, complex_g, comparison, chitchat
    enable_multimodal: bool = True
    max_tokens: int = 3072
    include_metadata: bool = True
    
    # 处理器特定配置
    processor_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.processor_configs is None:
            self.processor_configs = {}

@dataclass
class UnifiedInput:
    """统一LLM输入数据结构"""
    # 基础信息
    query: str
    processor_type: str
    
    # 文本上下文
    text_context: Optional[str] = None
    formatted_text: Optional[str] = None
    
    # 多模态信息
    multimodal_context: Optional[MultimodalContext] = None
    graph_embedding: Optional[List[float]] = None
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    # 处理器特定数据
    processor_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def has_multimodal_data(self) -> bool:
        """检查是否包含多模态数据"""
        return (self.multimodal_context is not None or 
                self.graph_embedding is not None)
    
    def get_text_content(self) -> str:
        """获取文本内容"""
        if self.formatted_text:
            return self.formatted_text
        elif self.text_context:
            return self.text_context
        elif self.multimodal_context:
            return self.multimodal_context.text_context
        else:
            return ""
    
    def get_graph_embedding(self) -> Optional[List[float]]:
        """获取图嵌入"""
        if self.graph_embedding:
            return self.graph_embedding
        elif self.multimodal_context and self.multimodal_context.graph_embedding:
            return self.multimodal_context.graph_embedding
        else:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'query': self.query,
            'processor_type': self.processor_type,
            'text_context': self.text_context,
            'formatted_text': self.formatted_text,
            'has_multimodal': self.has_multimodal_data(),
            'graph_embedding_dim': len(self.get_graph_embedding()) if self.get_graph_embedding() else 0,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'processor_data_keys': list(self.processor_data.keys()) if self.processor_data else []
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        multimodal_info = "有" if self.has_multimodal_data() else "无"
        text_len = len(self.get_text_content())
        return f"UnifiedInput(processor={self.processor_type}, text_len={text_len}, multimodal={multimodal_info})"

class InputRouter:
    """LLM输入路由器"""
    
    def __init__(self, config: LLMInputConfig):
        """初始化输入路由器"""
        self.config = config
        self.processor_handlers = {
            'direct': self._handle_direct_processor,
            'simple_g': self._handle_simple_g_processor,
            'complex_g': self._handle_complex_g_processor,
            'comparison': self._handle_comparison_processor,
            'chitchat': self._handle_chitchat_processor
        }
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'by_processor': {},
            'multimodal_count': 0,
            'text_only_count': 0
        }
        
        logger.info(f"🚦 InputRouter初始化完成，支持处理器: {list(self.processor_handlers.keys())}")
    
    def route_processor_output(self, processor_output: Dict[str, Any], query: str) -> UnifiedInput:
        """路由处理器输出到统一输入格式
        
        Args:
            processor_output: RAG处理器的输出
            query: 用户查询
            
        Returns:
            UnifiedInput: 统一格式的LLM输入
        """
        try:
            # 确定处理器类型
            processor_type = self._determine_processor_type(processor_output)
            
            # 选择对应的处理函数
            if processor_type in self.processor_handlers:
                handler = self.processor_handlers[processor_type]
                unified_input = handler(processor_output, query)
            else:
                # 使用默认处理器
                logger.warning(f"⚠️ 未识别的处理器类型: {processor_type}，使用默认处理")
                unified_input = self._handle_default_processor(processor_output, query)
            
            # 更新统计信息
            self._update_stats(unified_input)
            
            logger.info(f"🎯 输入路由完成: {unified_input}")
            return unified_input
            
        except Exception as e:
            logger.error(f"❌ 输入路由失败: {str(e)}")
            # 创建错误回退输入
            return self._create_fallback_input(processor_output, query, str(e))
    
    def _determine_processor_type(self, processor_output: Dict[str, Any]) -> str:
        """确定处理器类型"""
        # 方法1：从输出中直接获取
        if 'processor_type' in processor_output:
            return processor_output['processor_type']
        
        # 方法2：从元数据中获取
        metadata = processor_output.get('metadata', {})
        if 'processor_name' in metadata:
            processor_name = metadata['processor_name']
            # 映射处理器名称
            if 'direct' in processor_name.lower():
                return 'direct'
            elif 'simple' in processor_name.lower():
                return 'simple_g'
            elif 'complex' in processor_name.lower():
                return 'complex_g'
            elif 'comparison' in processor_name.lower():
                return 'comparison'
            elif 'chitchat' in processor_name.lower():
                return 'chitchat'
        
        # 方法3：基于输出结构推断
        if 'multimodal_context' in processor_output:
            return 'complex_g'
        elif 'graph' in processor_output and 'textual_context' in processor_output:
            return 'simple_g'
        elif 'comparison_result' in processor_output:
            return 'comparison'
        else:
            return 'direct'
    
    def _handle_direct_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理Direct处理器输出"""
        text_content = ""
        
        # 提取文本内容
        if 'textual_context' in output:
            if isinstance(output['textual_context'], dict):
                text_content = output['textual_context'].get('formatted_text', '')
            else:
                text_content = str(output['textual_context'])
        
        return UnifiedInput(
            query=query,
            processor_type='direct',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'direct',
                'has_graph_data': False,
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_simple_g_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理Simple G处理器输出"""
        text_content = ""
        
        # 提取文本内容
        textual_context = output.get('textual_context', {})
        if isinstance(textual_context, dict):
            text_content = textual_context.get('formatted_text', '')
        else:
            text_content = str(textual_context)
        
        return UnifiedInput(
            query=query,
            processor_type='simple_g',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'simple_graph',
                'has_graph_data': 'graph' in output,
                'graph_nodes': len(output.get('graph', {}).get('nodes', [])),
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_complex_g_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理Complex G处理器输出"""
        # 检查是否是增强模式输出
        if 'multimodal_context' in output:
            return self._handle_enhanced_mode_output(output, query)
        else:
            return self._handle_traditional_mode_output(output, query)
    
    def _handle_enhanced_mode_output(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理增强模式输出"""
        multimodal_context = output.get('multimodal_context')
        graph_embedding = None
        
        # 提取图嵌入
        if 'graph_embedding' in output:
            graph_embed_data = output['graph_embedding']
            if isinstance(graph_embed_data, dict):
                graph_embedding = graph_embed_data.get('embedding')
        
        # 提取文本内容
        text_content = ""
        if multimodal_context and hasattr(multimodal_context, 'text_context'):
            text_content = multimodal_context.text_context
        elif 'traditional_result' in output:
            traditional = output['traditional_result']
            if 'textual_context' in traditional:
                textual_context = traditional['textual_context']
                if isinstance(textual_context, dict):
                    text_content = textual_context.get('formatted_text', '')
                else:
                    text_content = str(textual_context)
        
        return UnifiedInput(
            query=query,
            processor_type='complex_g',
            text_context=text_content,
            formatted_text=text_content,
            multimodal_context=multimodal_context,
            graph_embedding=graph_embedding,
            metadata={
                'processing_mode': 'enhanced',
                'has_graph_embedding': graph_embedding is not None,
                'multimodal_fusion': True,
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_traditional_mode_output(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理传统模式输出"""
        text_content = ""
        
        # 提取文本内容
        textual_context = output.get('textual_context', {})
        if isinstance(textual_context, dict):
            text_content = textual_context.get('formatted_text', '')
        else:
            text_content = str(textual_context)
        
        return UnifiedInput(
            query=query,
            processor_type='complex_g',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'traditional',
                'has_graph_data': 'graph' in output,
                'multimodal_fusion': False,
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_comparison_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理Comparison处理器输出"""
        text_content = ""
        
        # 提取比较结果
        if 'comparison_result' in output:
            comparison = output['comparison_result']
            if isinstance(comparison, dict):
                text_content = comparison.get('formatted_text', str(comparison))
            else:
                text_content = str(comparison)
        elif 'textual_context' in output:
            textual_context = output['textual_context']
            if isinstance(textual_context, dict):
                text_content = textual_context.get('formatted_text', '')
            else:
                text_content = str(textual_context)
        
        return UnifiedInput(
            query=query,
            processor_type='comparison',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'comparison',
                'comparison_type': output.get('comparison_type', 'unknown'),
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_chitchat_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """处理Chitchat处理器输出"""
        text_content = ""
        
        # 提取闲聊内容
        if 'response' in output:
            text_content = str(output['response'])
        elif 'textual_context' in output:
            textual_context = output['textual_context']
            if isinstance(textual_context, dict):
                text_content = textual_context.get('formatted_text', '')
            else:
                text_content = str(textual_context)
        
        return UnifiedInput(
            query=query,
            processor_type='chitchat',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'chitchat',
                'conversational': True,
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _handle_default_processor(self, output: Dict[str, Any], query: str) -> UnifiedInput:
        """默认处理器处理"""
        # 尝试提取任何可用的文本内容
        text_content = ""
        
        # 尝试多种方式提取文本
        if 'textual_context' in output:
            textual_context = output['textual_context']
            if isinstance(textual_context, dict):
                text_content = textual_context.get('formatted_text', str(textual_context))
            else:
                text_content = str(textual_context)
        elif 'response' in output:
            text_content = str(output['response'])
        elif 'result' in output:
            text_content = str(output['result'])
        else:
            # 使用整个输出的字符串表示
            text_content = json.dumps(output, ensure_ascii=False, indent=2)
        
        return UnifiedInput(
            query=query,
            processor_type='unknown',
            text_context=text_content,
            formatted_text=text_content,
            metadata={
                'processing_mode': 'default',
                'fallback': True,
                **output.get('metadata', {})
            },
            processor_data=output
        )
    
    def _create_fallback_input(self, output: Dict[str, Any], query: str, error: str) -> UnifiedInput:
        """创建错误回退输入"""
        return UnifiedInput(
            query=query,
            processor_type='error',
            text_context=f"处理错误: {error}",
            formatted_text=f"无法处理输入数据，错误信息: {error}",
            metadata={
                'processing_mode': 'error',
                'error': error,
                'fallback': True
            },
            processor_data=output
        )
    
    def _update_stats(self, unified_input: UnifiedInput):
        """更新统计信息"""
        self.stats['total_processed'] += 1
        
        processor_type = unified_input.processor_type
        if processor_type not in self.stats['by_processor']:
            self.stats['by_processor'][processor_type] = 0
        self.stats['by_processor'][processor_type] += 1
        
        if unified_input.has_multimodal_data():
            self.stats['multimodal_count'] += 1
        else:
            self.stats['text_only_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_processed': self.stats['total_processed'],
            'processor_distribution': self.stats['by_processor'].copy(),
            'multimodal_ratio': self.stats['multimodal_count'] / max(1, self.stats['total_processed']),
            'text_only_ratio': self.stats['text_only_count'] / max(1, self.stats['total_processed']),
            'supported_processors': list(self.processor_handlers.keys())
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'by_processor': {},
            'multimodal_count': 0,
            'text_only_count': 0
        }

# =============================================================================
# 工厂函数
# =============================================================================

def create_input_router(config: Optional[LLMInputConfig] = None) -> InputRouter:
    """创建输入路由器"""
    if config is None:
        config = LLMInputConfig(
            processor_type='auto',
            enable_multimodal=True,
            max_tokens=3072,
            include_metadata=True
        )
    
    return InputRouter(config)

# =============================================================================
# 测试和演示
# =============================================================================

if __name__ == "__main__":
    # 创建测试路由器
    router = create_input_router()
    
    # 测试不同类型的处理器输出
    test_cases = [
        # Direct处理器输出
        {
            'processor_output': {
                'textual_context': {'formatted_text': '科比是洛杉矶湖人队的传奇球员。'},
                'metadata': {'processor_name': 'direct_processor'}
            },
            'query': '科比是谁？'
        },
        
        # Complex G处理器增强模式输出
        {
            'processor_output': {
                'mode': 'enhanced',
                'multimodal_context': None,  # 实际使用中会是MultimodalContext对象
                'graph_embedding': {'embedding': [0.1, 0.2, 0.3]},
                'metadata': {'processor_name': 'complex_g_processor'}
            },
            'query': '科比和詹姆斯的关系？'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 测试案例 {i+1}:")
        unified_input = router.route_processor_output(
            test_case['processor_output'], 
            test_case['query']
        )
        print(f"输入类型: {unified_input.processor_type}")
        print(f"多模态数据: {'有' if unified_input.has_multimodal_data() else '无'}")
        print(f"文本长度: {len(unified_input.get_text_content())}")
    
    # 显示统计信息
    print(f"\n📊 路由器统计: {router.get_stats()}")
