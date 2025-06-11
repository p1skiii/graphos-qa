"""
直接查询处理器 (Direct Processor)
用于处理 ATTRIBUTE_QUERY 类型的查询
适用于直接的属性查询，如"科比多少岁"、"湖人队主场在哪里"
"""
from typing import Dict, Any, Optional, List
import logging
from .base_processor import BaseProcessor, ProcessorUtils
from app.rag.components import ProcessorDefaultConfigs

logger = logging.getLogger(__name__)

class DirectProcessor(BaseProcessor):
    """直接查询处理器"""
    
    def __init__(self, config: Optional[ProcessorDefaultConfigs] = None):
        """初始化直接查询处理器"""
        if config is None:
            config = ProcessorDefaultConfigs.get_direct_processor_config()
        
        super().__init__(config)
        logger.info(f"🎯 创建直接查询处理器")
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """处理直接查询"""
        logger.info(f"🔍 直接查询处理: {query}")
        
        try:
            # 1. 实体提取和查询分析
            query_analysis = self._analyze_query(query, context)
            
            # 2. 检索相关节点 (使用关键词检索，更适合直接查询)
            retrieved_nodes = self.retriever.retrieve(
                query, 
                top_k=min(8, self.config.max_tokens // 200)  # 根据token限制调整
            )
            
            if not retrieved_nodes:
                return self._create_empty_result(query, "未找到相关实体")
            
            # 3. 构建简单子图 (直接查询通常不需要复杂图结构)
            seed_nodes = [node['node_id'] for node in retrieved_nodes[:5]]
            subgraph = self.graph_builder.build_subgraph(seed_nodes, query)
            
            if not ProcessorUtils.validate_subgraph(subgraph):
                return self._create_fallback_result(query, retrieved_nodes)
            
            # 4. 文本化 (使用紧凑格式)
            contextualized_text = self.textualizer.textualize(subgraph, query)
            
            # 5. 限制token数量
            final_text = ProcessorUtils.limit_tokens(
                contextualized_text, 
                self.config.max_tokens
            )
            
            # 6. 构建结果
            result = {
                'success': True,
                'query': query,
                'context': context or {},
                'query_analysis': query_analysis,
                'retrieved_nodes_count': len(retrieved_nodes),
                'subgraph_summary': {
                    'nodes': subgraph['node_count'],
                    'edges': subgraph['edge_count'],
                    'algorithm': subgraph.get('algorithm', 'unknown')
                },
                'contextualized_text': final_text,
                'processing_strategy': 'direct_attribute_query',
                'confidence': self._calculate_confidence(retrieved_nodes, subgraph)
            }
            
            logger.info(f"✅ 直接查询处理完成，节点数: {len(retrieved_nodes)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 直接查询处理失败: {str(e)}")
            raise
    
    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析查询特征"""
        analysis = {
            'query_type': 'attribute_query',
            'entities': ProcessorUtils.extract_entities_from_query(query),
            'intent': self._detect_attribute_intent(query),
            'direct_answer_expected': True,
            'complexity': 'simple'
        }
        
        # 分析上下文
        if context:
            analysis['context_entities'] = context.get('entities', [])
            analysis['previous_queries'] = context.get('query_history', [])
        
        return analysis
    
    def _detect_attribute_intent(self, query: str) -> Dict[str, Any]:
        """检测属性查询意图"""
        query_lower = query.lower()
        
        intent = {
            'attribute_type': 'unknown',
            'question_words': [],
            'target_entity': None
        }
        
        # 检测问词
        question_words = ['多少', '多大', '什么', '哪里', '哪个', '谁', '何时', '怎么']
        for word in question_words:
            if word in query:
                intent['question_words'].append(word)
        
        # 检测属性类型
        if any(word in query for word in ['年龄', '多大', '岁']):
            intent['attribute_type'] = 'age'
        elif any(word in query for word in ['身高', '多高']):
            intent['attribute_type'] = 'height'
        elif any(word in query for word in ['体重', '多重']):
            intent['attribute_type'] = 'weight'
        elif any(word in query for word in ['球队', '效力', '哪个队']):
            intent['attribute_type'] = 'team'
        elif any(word in query for word in ['位置', '打什么位置']):
            intent['attribute_type'] = 'position'
        elif any(word in query for word in ['得分', '场均得分']):
            intent['attribute_type'] = 'scoring'
        
        return intent
    
    def _create_empty_result(self, query: str, reason: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'attribute_query'},
            'retrieved_nodes_count': 0,
            'subgraph_summary': {'nodes': 0, 'edges': 0, 'algorithm': 'none'},
            'contextualized_text': f"抱歉，{reason}。请尝试更具体的查询。",
            'processing_strategy': 'direct_attribute_query',
            'confidence': 0.0,
            'empty_reason': reason
        }
    
    def _create_fallback_result(self, query: str, retrieved_nodes: List[Dict]) -> Dict[str, Any]:
        """创建回退结果（当子图构建失败时）"""
        # 直接使用检索到的节点信息
        fallback_text = self._format_retrieved_nodes(retrieved_nodes, query)
        
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'attribute_query'},
            'retrieved_nodes_count': len(retrieved_nodes),
            'subgraph_summary': {'nodes': len(retrieved_nodes), 'edges': 0, 'algorithm': 'fallback'},
            'contextualized_text': fallback_text,
            'processing_strategy': 'direct_attribute_query_fallback',
            'confidence': 0.7
        }
    
    def _format_retrieved_nodes(self, nodes: List[Dict], query: str) -> str:
        """格式化检索到的节点信息"""
        if not nodes:
            return "未找到相关信息。"
        
        text_parts = ["根据查询找到以下相关信息："]
        
        # 按类型分组
        players = [n for n in nodes if n.get('type') == 'player']
        teams = [n for n in nodes if n.get('type') == 'team']
        
        # 格式化球员信息
        if players:
            text_parts.append("\n球员信息：")
            for player in players[:3]:  # 最多显示3个
                name = player.get('name', '未知球员')
                properties = player.get('properties', {})
                
                info_parts = [f"- {name}"]
                if properties.get('age'):
                    info_parts.append(f"年龄{properties['age']}岁")
                
                text_parts.append(" ".join(info_parts))
        
        # 格式化球队信息
        if teams:
            text_parts.append("\n球队信息：")
            for team in teams[:3]:  # 最多显示3个
                name = team.get('name', '未知球队')
                text_parts.append(f"- {name}")
        
        return "\n".join(text_parts)
    
    def _calculate_confidence(self, retrieved_nodes: List[Dict], subgraph: Dict[str, Any]) -> float:
        """计算结果置信度"""
        if not retrieved_nodes:
            return 0.0
        
        # 基础置信度
        base_confidence = 0.6
        
        # 根据检索节点数量调整
        node_bonus = min(0.2, len(retrieved_nodes) * 0.05)
        
        # 根据相似度调整
        similarities = [node.get('similarity', 0) for node in retrieved_nodes]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            similarity_bonus = avg_similarity * 0.2
        else:
            similarity_bonus = 0
        
        # 根据子图质量调整
        subgraph_bonus = 0
        if subgraph and subgraph.get('node_count', 0) > 0:
            subgraph_bonus = min(0.1, subgraph['node_count'] * 0.02)
        
        total_confidence = base_confidence + node_bonus + similarity_bonus + subgraph_bonus
        return min(1.0, total_confidence)

# =============================================================================
# 工厂函数
# =============================================================================

def create_direct_processor(custom_config: Optional[Dict[str, Any]] = None) -> DirectProcessor:
    """创建直接查询处理器实例"""
    if custom_config:
        # 如果提供了自定义配置，更新默认配置
        config = ProcessorDefaultConfigs.get_direct_processor_config()
        
        # 更新检索器配置
        if 'retriever' in custom_config:
            retriever_updates = custom_config['retriever']
            config.retriever_config.config.update(retriever_updates)
        
        # 更新图构建器配置
        if 'graph_builder' in custom_config:
            builder_updates = custom_config['graph_builder']
            config.graph_builder_config.config.update(builder_updates)
        
        # 更新文本化器配置
        if 'textualizer' in custom_config:
            textualizer_updates = custom_config['textualizer']
            config.textualizer_config.config.update(textualizer_updates)
        
        # 更新处理器级别配置
        if 'cache_enabled' in custom_config:
            config.cache_enabled = custom_config['cache_enabled']
        if 'cache_ttl' in custom_config:
            config.cache_ttl = custom_config['cache_ttl']
        if 'max_tokens' in custom_config:
            config.max_tokens = custom_config['max_tokens']
        
        return DirectProcessor(config)
    else:
        return DirectProcessor()
