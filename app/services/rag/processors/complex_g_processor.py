"""
复杂图处理器 (Complex Graph Processor)
支持双模式的RAG处理器：传统模式和增强模式
- 传统模式：基于文本的检索和图谱构建
- 增强模式：结合GraphEncoder的多模态处理

基于G-Retriever论文的多模态融合方法
"""
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from app.rag.processors.base_processor import BaseProcessor
from app.rag.components import ProcessorConfig
from app.rag.components.graph_encoder import GraphEncoder, MultimodalContext, create_graph_encoder

logger = logging.getLogger(__name__)

@dataclass
class ComplexGProcessorConfig(ProcessorConfig):
    """ComplexGProcessor配置"""
    # 处理模式配置
    use_enhanced_mode: bool = False  # 是否使用增强模式
    enable_multimodal_fusion: bool = False  # 是否启用多模态融合
    
    # GraphEncoder配置
    graph_encoder_enabled: bool = False
    graph_encoder_config: Optional[Dict[str, Any]] = None
    
    # 增强模式特定配置
    min_graph_nodes: int = 3  # 最小图节点数，低于此值使用传统模式
    fusion_strategy: str = "concatenate"  # 融合策略: concatenate, weighted, attention
    
    # 性能配置
    fallback_to_traditional: bool = True  # 增强模式失败时是否回退到传统模式

class ComplexGProcessor(BaseProcessor):
    """复杂图处理器 - 支持传统和增强双模式"""
    
    def __init__(self, config: ComplexGProcessorConfig):
        super().__init__(config)
        self.complex_config = config
        
        # GraphEncoder组件
        self.graph_encoder = None
        
        # 模式状态
        self.current_mode = "traditional"  # traditional | enhanced
        self.mode_switch_count = 0
        
        # 统计信息
        self.enhanced_stats = {
            'traditional_mode_count': 0,
            'enhanced_mode_count': 0,
            'mode_switches': 0,
            'graph_encoding_time': 0.0,
            'multimodal_fusion_time': 0.0
        }
    
    def initialize(self) -> bool:
        """初始化处理器"""
        try:
            # 调用基类初始化
            if not super().initialize():
                return False
            
            # 初始化GraphEncoder（如果启用）
            if self.complex_config.graph_encoder_enabled:
                self._initialize_graph_encoder()
            
            # 确定默认模式
            self.current_mode = "enhanced" if self.complex_config.use_enhanced_mode else "traditional"
            
            logger.info(f"✅ ComplexGProcessor初始化完成，默认模式: {self.current_mode}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ComplexGProcessor初始化失败: {str(e)}")
            return False
    
    def _initialize_graph_encoder(self):
        """初始化GraphEncoder组件"""
        try:
            graph_config = self.complex_config.graph_encoder_config or {}
            self.graph_encoder = create_graph_encoder(graph_config)
            
            # 调用GraphEncoder的初始化方法
            if self.graph_encoder.initialize():
                logger.info("✅ GraphEncoder组件初始化完成")
            else:
                logger.error("❌ GraphEncoder初始化失败")
                self.graph_encoder = None
                if not self.complex_config.fallback_to_traditional:
                    raise RuntimeError("GraphEncoder初始化失败")
            
        except Exception as e:
            logger.error(f"❌ GraphEncoder初始化失败: {str(e)}")
            self.graph_encoder = None
            if not self.complex_config.fallback_to_traditional:
                raise
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """具体处理逻辑 - 支持双模式"""
        try:
            # 确定处理模式
            processing_mode = self._determine_processing_mode(query, context)
            
            if processing_mode == "enhanced" and self.graph_encoder:
                return self._process_enhanced_mode(query, context)
            else:
                return self._process_traditional_mode(query, context)
                
        except Exception as e:
            logger.error(f"❌ ComplexGProcessor处理失败: {str(e)}")
            
            # 如果增强模式失败且允许回退，使用传统模式
            if self.current_mode == "enhanced" and self.complex_config.fallback_to_traditional:
                logger.warning("🔄 增强模式失败，回退到传统模式")
                return self._process_traditional_mode(query, context)
            
            raise
    
    def _determine_processing_mode(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """确定处理模式"""
        # 如果GraphEncoder未启用，使用传统模式
        if not self.graph_encoder:
            return "traditional"
        
        # 如果强制使用某种模式
        if context and context.get('force_mode'):
            return context['force_mode']
        
        # 基于配置和图复杂度决定
        if self.complex_config.use_enhanced_mode:
            # 检查是否满足增强模式的条件
            if self._should_use_enhanced_mode(query, context):
                return "enhanced"
        
        return "traditional"
    
    def _should_use_enhanced_mode(self, query: str, context: Optional[Dict[str, Any]]) -> bool:
        """判断是否应该使用增强模式"""
        try:
            # 基本检查：是否有足够的图数据
            if context and 'graph_data' in context:
                graph_data = context['graph_data']
                if isinstance(graph_data, dict) and 'nodes' in graph_data:
                    node_count = len(graph_data['nodes'])
                    return node_count >= self.complex_config.min_graph_nodes
            
            # 可以添加更多智能判断逻辑
            # 例如：查询复杂度、实体数量等
            
            return True  # 默认使用增强模式（如果启用）
            
        except Exception as e:
            logger.warning(f"⚠️ 模式判断失败，使用传统模式: {str(e)}")
            return False
    
    def _process_traditional_mode(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """传统模式处理"""
        start_time = time.time()
        
        try:
            self.enhanced_stats['traditional_mode_count'] += 1
            logger.info(f"🔄 使用传统模式处理: {query[:50]}...")
            
            # 1. 文档检索
            retrieval_result = self.retriever.retrieve(query, context)
            
            # 2. 图谱构建
            graph_result = self.graph_builder.build_graph(
                query, 
                retrieval_result, 
                context
            )
            
            # 3. 文本化
            textual_result = self.textualizer.textualize(
                query,
                graph_result,
                context
            )
            
            # 构建结果
            result = {
                'success': True,
                'mode': 'traditional',
                'query': query,
                'retrieval': retrieval_result,
                'graph': graph_result,
                'textual_context': textual_result,
                'processing_time': time.time() - start_time
            }
            
            logger.info(f"✅ 传统模式处理完成，耗时: {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ 传统模式处理失败: {str(e)}")
            raise
    
    def _process_enhanced_mode(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """增强模式处理 - 结合GraphEncoder"""
        start_time = time.time()
        
        try:
            self.enhanced_stats['enhanced_mode_count'] += 1
            logger.info(f"🚀 使用增强模式处理: {query[:50]}...")
            
            # 1. 执行传统流程获取基础结果
            traditional_result = self._process_traditional_mode(query, context)
            
            # 2. 图编码 - 生成图嵌入
            graph_encoding_start = time.time()
            graph_embedding = self._encode_graph_data(traditional_result['graph'])
            graph_encoding_time = time.time() - graph_encoding_start
            self.enhanced_stats['graph_encoding_time'] += graph_encoding_time
            
            # 3. 增强图嵌入的语义理解
            enhanced_graph_info = self._enhance_graph_semantics(
                graph_embedding, 
                traditional_result['graph'], 
                query
            )
            
            # 4. 多模态融合
            fusion_start = time.time()
            multimodal_context = self._create_multimodal_context(
                traditional_result['textual_context'],
                enhanced_graph_info,
                query
            )
            fusion_time = time.time() - fusion_start
            self.enhanced_stats['multimodal_fusion_time'] += fusion_time
            
            # 构建增强结果
            result = {
                'success': True,
                'mode': 'enhanced',
                'query': query,
                'traditional_result': traditional_result,
                'graph_embedding': enhanced_graph_info,
                'multimodal_context': multimodal_context,
                'enhanced_metadata': {
                    'graph_summary': enhanced_graph_info.get('semantic_summary', ''),
                    'fusion_strategy': self.complex_config.fusion_strategy,
                    'llm_ready': True
                },
                'enhanced_metrics': {
                    'graph_encoding_time': graph_encoding_time,
                    'fusion_time': fusion_time,
                    'total_time': time.time() - start_time
                }
            }
            
            logger.info(f"✅ 增强模式处理完成，耗时: {result['enhanced_metrics']['total_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ 增强模式处理失败: {str(e)}")
            raise
    
    def _enhance_graph_semantics(self, graph_embedding: Optional[Dict[str, Any]], 
                                graph_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """增强图嵌入的语义理解"""
        try:
            if not graph_embedding or not graph_embedding.get('encoding_success'):
                return graph_embedding or {}
            
            # 基础信息
            enhanced_info = graph_embedding.copy()
            
            # 添加语义摘要
            semantic_summary = self._generate_graph_summary(graph_data, query)
            enhanced_info['semantic_summary'] = semantic_summary
            
            # 添加实体关系分析
            entity_analysis = self._analyze_entities_and_relations(graph_data)
            enhanced_info['entity_analysis'] = entity_analysis
            
            # 添加查询相关性分析
            relevance_info = self._analyze_query_relevance(graph_data, query)
            enhanced_info['query_relevance'] = relevance_info
            
            return enhanced_info
            
        except Exception as e:
            logger.warning(f"⚠️ 图语义增强失败: {str(e)}")
            return graph_embedding or {}
    
    def _generate_graph_summary(self, graph_data: Dict[str, Any], query: str) -> str:
        """生成图谱摘要"""
        try:
            summary_parts = []
            
            # 分析图结构
            if 'graph_structure' in graph_data:
                structure = graph_data['graph_structure']
                node_count = structure.get('num_nodes', 0)
                edge_count = structure.get('num_edges', 0)
                summary_parts.append(f"包含{node_count}个实体和{edge_count}个关系")
            
            # 分析实体类型
            if 'metadata' in graph_data:
                metadata = graph_data['metadata']
                if 'entity_types' in metadata and metadata['entity_types']:
                    types = metadata['entity_types'][:3]  # 取前3个类型
                    summary_parts.append(f"主要实体类型：{', '.join(types)}")
            
            # 分析关键节点
            if 'key_nodes' in graph_data:
                key_nodes = graph_data['key_nodes'][:3]  # 取前3个关键节点
                if key_nodes:
                    summary_parts.append(f"关键实体：{', '.join(key_nodes)}")
            
            base_summary = "图谱" + "，".join(summary_parts) if summary_parts else "图谱结构信息"
            
            # 添加查询相关性
            if query:
                base_summary += f"，与查询'{query[:20]}...'相关"
            
            return base_summary
            
        except Exception as e:
            logger.warning(f"⚠️ 图谱摘要生成失败: {str(e)}")
            return "图谱结构特征"
    
    def _analyze_entities_and_relations(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析实体和关系"""
        try:
            analysis = {
                'entity_count': 0,
                'relation_count': 0,
                'entity_types': [],
                'relation_types': []
            }
            
            # 分析节点
            if 'nodes' in graph_data:
                nodes = graph_data['nodes']
                analysis['entity_count'] = len(nodes)
                
                # 收集实体类型
                entity_types = set()
                for node in nodes:
                    if 'label' in node:
                        entity_types.add(node['label'])
                analysis['entity_types'] = list(entity_types)
            
            # 分析边
            if 'edges' in graph_data:
                edges = graph_data['edges']
                analysis['relation_count'] = len(edges)
                
                # 收集关系类型
                relation_types = set()
                for edge in edges:
                    if 'relation' in edge:
                        relation_types.add(edge['relation'])
                analysis['relation_types'] = list(relation_types)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ 实体关系分析失败: {str(e)}")
            return {}
    
    def _analyze_query_relevance(self, graph_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """分析查询相关性"""
        try:
            relevance_info = {
                'query_entities': [],
                'matched_nodes': [],
                'relevance_score': 0.0
            }
            
            # 简单的查询实体提取（实际应用中可能需要更复杂的NER）
            query_lower = query.lower()
            query_entities = []
            
            # 从图中寻找匹配的实体
            if 'nodes' in graph_data:
                matched_nodes = []
                for node in graph_data['nodes']:
                    node_name = node.get('name', '').lower()
                    if node_name and node_name in query_lower:
                        matched_nodes.append(node['id'])
                        query_entities.append(node_name)
                
                relevance_info['query_entities'] = query_entities
                relevance_info['matched_nodes'] = matched_nodes
                
                # 计算相关性分数
                if len(matched_nodes) > 0:
                    relevance_info['relevance_score'] = min(1.0, len(matched_nodes) / 3.0)
            
            return relevance_info
            
        except Exception as e:
            logger.warning(f"⚠️ 查询相关性分析失败: {str(e)}")
            return {}
    
    def _encode_graph_data(self, graph_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """编码图数据为向量表示"""
        try:
            if not self.graph_encoder:
                logger.warning("⚠️ GraphEncoder未初始化")
                return None
            
            # 使用GraphEncoder编码图数据
            encoding_result = self.graph_encoder.encode_graph(graph_data)
            
            return {
                'embedding': encoding_result.get('embedding'),
                'node_embeddings': encoding_result.get('node_embeddings'),
                'encoding_metadata': encoding_result.get('metadata', {}),
                'encoding_success': True
            }
            
        except Exception as e:
            logger.error(f"❌ 图编码失败: {str(e)}")
            return {
                'embedding': None,
                'encoding_success': False,
                'error': str(e)
            }
    
    def _create_multimodal_context(
        self, 
        textual_context: Dict[str, Any], 
        graph_embedding: Optional[Dict[str, Any]], 
        query: str
    ) -> MultimodalContext:
        """创建多模态上下文"""
        try:
            # 提取文本表示
            text_content = ""
            if isinstance(textual_context, dict):
                text_content = textual_context.get('formatted_text', '')
                if not text_content:
                    text_content = str(textual_context.get('content', ''))
            
            # 提取图嵌入
            graph_embed = None
            if graph_embedding and graph_embedding.get('encoding_success'):
                graph_embed = graph_embedding.get('embedding')
            
            # 构建增强的元数据
            enhanced_metadata = {
                'query': query,
                'fusion_strategy': self.complex_config.fusion_strategy,
                'creation_time': time.time(),
                'graph_encoding_success': graph_embedding.get('encoding_success', False) if graph_embedding else False
            }
            
            # 添加图语义信息到元数据
            if graph_embedding:
                if 'semantic_summary' in graph_embedding:
                    enhanced_metadata['graph_summary'] = graph_embedding['semantic_summary']
                if 'entity_analysis' in graph_embedding:
                    enhanced_metadata['entity_analysis'] = graph_embedding['entity_analysis']
                if 'query_relevance' in graph_embedding:
                    enhanced_metadata['query_relevance'] = graph_embedding['query_relevance']
            
            # 创建MultimodalContext
            multimodal_context = MultimodalContext(
                text_context=text_content,
                graph_embedding=graph_embed,
                metadata=enhanced_metadata
            )
            
            return multimodal_context
            
        except Exception as e:
            logger.error(f"❌ 多模态上下文创建失败: {str(e)}")
            # 返回仅包含文本的上下文
            return MultimodalContext(
                text_context=str(textual_context),
                graph_embedding=None,
                metadata={'error': str(e), 'fallback': True}
            )
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """获取增强统计信息"""
        base_stats = self.get_stats()
        
        enhanced_info = {
            'processing_modes': self.enhanced_stats.copy(),
            'current_mode': self.current_mode,
            'graph_encoder_enabled': self.graph_encoder is not None,
            'multimodal_fusion_enabled': self.complex_config.enable_multimodal_fusion,
            'config': {
                'use_enhanced_mode': self.complex_config.use_enhanced_mode,
                'min_graph_nodes': self.complex_config.min_graph_nodes,
                'fusion_strategy': self.complex_config.fusion_strategy,
                'fallback_enabled': self.complex_config.fallback_to_traditional
            }
        }
        
        base_stats['enhanced_info'] = enhanced_info
        return base_stats
    
    def switch_mode(self, mode: str) -> bool:
        """手动切换处理模式"""
        if mode not in ['traditional', 'enhanced']:
            logger.error(f"❌ 无效的处理模式: {mode}")
            return False
        
        if mode == 'enhanced' and not self.graph_encoder:
            logger.error("❌ GraphEncoder未启用，无法切换到增强模式")
            return False
        
        old_mode = self.current_mode
        self.current_mode = mode
        self.mode_switch_count += 1
        self.enhanced_stats['mode_switches'] += 1
        
        logger.info(f"🔄 模式切换: {old_mode} → {mode}")
        return True
    
    def test_graph_encoder(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """测试GraphEncoder功能"""
        if not self.graph_encoder:
            return {'success': False, 'error': 'GraphEncoder未启用'}
        
        try:
            # 使用测试数据或创建简单测试图
            if not test_data:
                test_data = {
                    'nodes': [
                        {'id': 'player_1', 'label': 'Player', 'name': '梅西'},
                        {'id': 'team_1', 'label': 'Team', 'name': '巴塞罗那'}
                    ],
                    'edges': [
                        {'source': 'player_1', 'target': 'team_1', 'relation': 'plays_for'}
                    ]
                }
            
            result = self.graph_encoder.encode_graph(test_data)
            
            return {
                'success': True,
                'test_result': result,
                'encoder_info': {
                    'model_type': type(self.graph_encoder).__name__,
                    'has_model': hasattr(self.graph_encoder, 'model')
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# =============================================================================
# 工厂函数
# =============================================================================

def create_complex_g_processor(config_dict: Dict[str, Any]) -> ComplexGProcessor:
    """创建ComplexGProcessor实例"""
    
    # 转换配置
    config = ComplexGProcessorConfig(
        processor_name=config_dict.get('processor_name', 'complex_g_processor'),
        cache_enabled=config_dict.get('cache_enabled', True),
        cache_ttl=config_dict.get('cache_ttl', 3600),
        max_tokens=config_dict.get('max_tokens', 4000),
        
        # ComplexG特定配置
        use_enhanced_mode=config_dict.get('use_enhanced_mode', False),
        enable_multimodal_fusion=config_dict.get('enable_multimodal_fusion', False),
        graph_encoder_enabled=config_dict.get('graph_encoder_enabled', False),
        graph_encoder_config=config_dict.get('graph_encoder_config', {}),
        min_graph_nodes=config_dict.get('min_graph_nodes', 3),
        fusion_strategy=config_dict.get('fusion_strategy', 'concatenate'),
        fallback_to_traditional=config_dict.get('fallback_to_traditional', True),
        
        # 组件配置
        retriever_config=config_dict.get('retriever_config'),
        graph_builder_config=config_dict.get('graph_builder_config'),
        textualizer_config=config_dict.get('textualizer_config')
    )
    
    return ComplexGProcessor(config)
