"""
简单图查询处理器 (Simple Graph Processor)
用于处理 SIMPLE_RELATION_QUERY 类型的查询
适用于简单的关系查询，如"科比在哪个球队"、"湖人队有哪些球员"
"""
from typing import Dict, Any, Optional, List
import logging
from .base_processor import BaseProcessor, ProcessorUtils
from app.rag.components import ProcessorDefaultConfigs

logger = logging.getLogger(__name__)

class SimpleGProcessor(BaseProcessor):
    """简单图查询处理器"""
    
    def __init__(self, config: Optional[ProcessorDefaultConfigs] = None):
        """初始化简单图查询处理器"""
        if config is None:
            config = ProcessorDefaultConfigs.get_simple_g_processor_config()
        
        super().__init__(config)
        logger.info(f"🌐 创建简单图查询处理器")
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """处理简单图查询"""
        logger.info(f"🔍 简单图查询处理: {query}")
        
        try:
            # 1. 查询分析
            query_analysis = self._analyze_query(query, context)
            
            # 2. 语义检索相关节点
            retrieved_nodes = self.retriever.retrieve(
                query, 
                top_k=min(10, self.config.max_tokens // 150)
            )
            
            if not retrieved_nodes:
                return self._create_empty_result(query, "未找到相关实体")
            
            # 3. 构建简单子图 (BFS扩展)
            seed_nodes = [node['node_id'] for node in retrieved_nodes[:6]]
            subgraph = self.graph_builder.build_subgraph(seed_nodes, query)
            
            if not ProcessorUtils.validate_subgraph(subgraph):
                return self._create_fallback_result(query, retrieved_nodes)
            
            # 4. 增强子图 (添加关系信息)
            enhanced_subgraph = self._enhance_subgraph(subgraph, query_analysis)
            
            # 5. 文本化 (使用模板格式，适合关系展示)
            contextualized_text = self.textualizer.textualize(enhanced_subgraph, query)
            
            # 6. 限制token数量
            final_text = ProcessorUtils.limit_tokens(
                contextualized_text, 
                self.config.max_tokens
            )
            
            # 7. 构建结果
            result = {
                'success': True,
                'query': query,
                'context': context or {},
                'query_analysis': query_analysis,
                'retrieved_nodes_count': len(retrieved_nodes),
                'subgraph_summary': {
                    'nodes': enhanced_subgraph['node_count'],
                    'edges': enhanced_subgraph['edge_count'],
                    'algorithm': enhanced_subgraph.get('algorithm', 'bfs'),
                    'relation_types': self._get_relation_types(enhanced_subgraph)
                },
                'contextualized_text': final_text,
                'processing_strategy': 'simple_graph_relation',
                'confidence': self._calculate_confidence(retrieved_nodes, enhanced_subgraph)
            }
            
            logger.info(f"✅ 简单图查询处理完成，节点数: {enhanced_subgraph['node_count']}, 边数: {enhanced_subgraph['edge_count']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 简单图查询处理失败: {str(e)}")
            raise
    
    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析简单图查询"""
        analysis = {
            'query_type': 'simple_relation_query',
            'entities': ProcessorUtils.extract_entities_from_query(query),
            'relation_intent': self._detect_relation_intent(query),
            'graph_complexity': 'simple',
            'expected_hops': 1  # 简单查询通常只需要1跳关系
        }
        
        # 分析上下文
        if context:
            analysis['context_entities'] = context.get('entities', [])
            analysis['context_relations'] = context.get('relations', [])
        
        return analysis
    
    def _detect_relation_intent(self, query: str) -> Dict[str, Any]:
        """检测关系查询意图"""
        query_lower = query.lower()
        
        intent = {
            'relation_type': 'unknown',
            'direction': 'bidirectional',  # forward, backward, bidirectional
            'target_entity_type': None,
            'relation_keywords': []
        }
        
        # 检测关系类型
        if any(word in query for word in ['效力', '在哪个队', '球队']):
            intent['relation_type'] = 'team_affiliation'
            intent['relation_keywords'].append('serve')
            
            if '球员' in query or any(name in query for name in ['科比', '詹姆斯', '乔丹']):
                intent['direction'] = 'forward'  # 球员 -> 球队
                intent['target_entity_type'] = 'team'
            elif '球队' in query or any(team in query for team in ['湖人', '公牛', '勇士']):
                intent['direction'] = 'backward'  # 球队 -> 球员
                intent['target_entity_type'] = 'player'
        
        elif any(word in query for word in ['队友', '一起打球', '同队']):
            intent['relation_type'] = 'teammate'
            intent['direction'] = 'bidirectional'
            intent['target_entity_type'] = 'player'
        
        elif any(word in query for word in ['对手', '交手', '比赛']):
            intent['relation_type'] = 'opponent'
            intent['direction'] = 'bidirectional'
        
        return intent
    
    def _enhance_subgraph(self, subgraph: Dict[str, Any], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """增强子图信息"""
        enhanced_subgraph = subgraph.copy()
        
        # 添加关系统计
        relation_stats = self._analyze_subgraph_relations(subgraph)
        enhanced_subgraph['relation_stats'] = relation_stats
        
        # 根据查询意图突出重要关系
        relation_intent = query_analysis.get('relation_intent', {})
        if relation_intent.get('relation_type') != 'unknown':
            enhanced_subgraph = self._highlight_relevant_relations(
                enhanced_subgraph, 
                relation_intent
            )
        
        return enhanced_subgraph
    
    def _analyze_subgraph_relations(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """分析子图中的关系"""
        edges = subgraph.get('edges', [])
        
        relation_stats = {
            'total_relations': len(edges),
            'relation_types': {},
            'entity_connections': {}
        }
        
        # 统计关系类型
        for edge in edges:
            relation = edge.get('relation', 'unknown')
            relation_stats['relation_types'][relation] = \
                relation_stats['relation_types'].get(relation, 0) + 1
        
        # 统计实体连接度
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            for entity in [source, target]:
                if entity:
                    relation_stats['entity_connections'][entity] = \
                        relation_stats['entity_connections'].get(entity, 0) + 1
        
        return relation_stats
    
    def _highlight_relevant_relations(self, subgraph: Dict[str, Any], 
                                    relation_intent: Dict[str, Any]) -> Dict[str, Any]:
        """突出显示相关关系"""
        highlighted_subgraph = subgraph.copy()
        
        target_relation = relation_intent.get('relation_type')
        if target_relation == 'team_affiliation':
            # 优先显示serve关系
            edges = highlighted_subgraph.get('edges', [])
            serve_edges = [e for e in edges if e.get('relation') == 'serve']
            other_edges = [e for e in edges if e.get('relation') != 'serve']
            
            # 重新排序，serve关系在前
            highlighted_subgraph['edges'] = serve_edges + other_edges
            highlighted_subgraph['primary_relation'] = 'serve'
        
        return highlighted_subgraph
    
    def _get_relation_types(self, subgraph: Dict[str, Any]) -> List[str]:
        """获取子图中的关系类型"""
        edges = subgraph.get('edges', [])
        relation_types = set()
        
        for edge in edges:
            relation = edge.get('relation', 'unknown')
            relation_types.add(relation)
        
        return list(relation_types)
    
    def _create_empty_result(self, query: str, reason: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'simple_relation_query'},
            'retrieved_nodes_count': 0,
            'subgraph_summary': {
                'nodes': 0, 
                'edges': 0, 
                'algorithm': 'none',
                'relation_types': []
            },
            'contextualized_text': f"抱歉，{reason}。请尝试提及具体的球员或球队名称。",
            'processing_strategy': 'simple_graph_relation',
            'confidence': 0.0,
            'empty_reason': reason
        }
    
    def _create_fallback_result(self, query: str, retrieved_nodes: List[Dict]) -> Dict[str, Any]:
        """创建回退结果"""
        fallback_text = self._format_simple_relations(retrieved_nodes, query)
        
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'simple_relation_query'},
            'retrieved_nodes_count': len(retrieved_nodes),
            'subgraph_summary': {
                'nodes': len(retrieved_nodes), 
                'edges': 0, 
                'algorithm': 'fallback',
                'relation_types': []
            },
            'contextualized_text': fallback_text,
            'processing_strategy': 'simple_graph_relation_fallback',
            'confidence': 0.7
        }
    
    def _format_simple_relations(self, nodes: List[Dict], query: str) -> str:
        """格式化简单关系信息"""
        if not nodes:
            return "未找到相关关系信息。"
        
        text_parts = ["根据查询找到以下关系信息："]
        
        # 按类型分组节点
        players = [n for n in nodes if n.get('type') == 'player']
        teams = [n for n in nodes if n.get('type') == 'team']
        
        # 如果同时有球员和球队，推断可能的关系
        if players and teams:
            text_parts.append("\n可能的效力关系：")
            for i, player in enumerate(players[:3]):
                for j, team in enumerate(teams[:2]):
                    if i < len(teams):  # 简单配对
                        player_name = player.get('name', '未知球员')
                        team_name = team.get('name', '未知球队')
                        text_parts.append(f"- {player_name} 可能与 {team_name} 相关")
        
        # 单独列出实体
        elif players:
            text_parts.append(f"\n相关球员：{', '.join([p.get('name', '未知') for p in players[:5]])}")
        elif teams:
            text_parts.append(f"\n相关球队：{', '.join([t.get('name', '未知') for t in teams[:5]])}")
        
        return "\n".join(text_parts)
    
    def _calculate_confidence(self, retrieved_nodes: List[Dict], subgraph: Dict[str, Any]) -> float:
        """计算结果置信度"""
        if not retrieved_nodes:
            return 0.0
        
        # 基础置信度
        base_confidence = 0.65
        
        # 根据检索质量调整
        similarities = [node.get('similarity', 0) for node in retrieved_nodes]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            similarity_bonus = avg_similarity * 0.15
        else:
            similarity_bonus = 0
        
        # 根据子图质量调整
        subgraph_bonus = 0
        if subgraph:
            node_count = subgraph.get('node_count', 0)
            edge_count = subgraph.get('edge_count', 0)
            
            if edge_count > 0:  # 有关系边是关键
                subgraph_bonus = min(0.2, edge_count * 0.05)
            elif node_count > 1:  # 至少有多个节点
                subgraph_bonus = 0.1
        
        total_confidence = base_confidence + similarity_bonus + subgraph_bonus
        return min(1.0, total_confidence)

# =============================================================================
# 工厂函数
# =============================================================================

def create_simple_g_processor(custom_config: Optional[Dict[str, Any]] = None) -> SimpleGProcessor:
    """创建简单图查询处理器实例"""
    if custom_config:
        config = ProcessorDefaultConfigs.get_simple_g_processor_config()
        
        # 更新配置逻辑（与DirectProcessor类似）
        if 'retriever' in custom_config:
            config.retriever_config.config.update(custom_config['retriever'])
        if 'graph_builder' in custom_config:
            config.graph_builder_config.config.update(custom_config['graph_builder'])
        if 'textualizer' in custom_config:
            config.textualizer_config.config.update(custom_config['textualizer'])
        
        for key in ['cache_enabled', 'cache_ttl', 'max_tokens']:
            if key in custom_config:
                setattr(config, key, custom_config[key])
        
        return SimpleGProcessor(config)
    else:
        return SimpleGProcessor()
