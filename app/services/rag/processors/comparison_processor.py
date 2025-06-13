"""
比较查询处理器 (Comparison Processor)
用于处理 COMPARATIVE_QUERY 类型的查询
适用于比较类查询，如"科比和詹姆斯谁更强"、"湖人和勇士哪个队更好"
"""
from typing import Dict, Any, Optional, List, Tuple
import logging
from .base_processor import BaseProcessor, ProcessorUtils
from app.rag.components import ProcessorDefaultConfigs

logger = logging.getLogger(__name__)

class ComparisonProcessor(BaseProcessor):
    """比较查询处理器"""
    
    def __init__(self, config: Optional[ProcessorDefaultConfigs] = None):
        """初始化比较查询处理器"""
        if config is None:
            config = ProcessorDefaultConfigs.get_comparison_processor_config()
        
        super().__init__(config)
        logger.info(f"⚖️ 创建比较查询处理器")
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """处理比较查询"""
        logger.info(f"🔍 比较查询处理: {query}")
        
        try:
            # 1. 比较查询分析
            query_analysis = self._analyze_comparison_query(query, context)
            
            # 2. 分别检索比较实体
            comparison_entities = query_analysis.get('comparison_entities', [])
            entity_nodes = self._retrieve_comparison_entities(query, comparison_entities)
            
            if not entity_nodes:
                return self._create_empty_result(query, "No comparable entities found")
            
            # 3. 构建比较子图
            all_seed_nodes = []
            for entity_group in entity_nodes.values():
                all_seed_nodes.extend([node['node_id'] for node in entity_group[:3]])
            
            subgraph = self.graph_builder.build_subgraph(all_seed_nodes, query)
            
            if not ProcessorUtils.validate_subgraph(subgraph):
                return self._create_fallback_result(query, entity_nodes)
            
            # 4. 执行比较分析
            comparison_analysis = self._perform_comparison_analysis(
                subgraph, entity_nodes, query_analysis
            )
            
            # 5. 生成比较文本
            comparison_text = self._generate_comparison_text(
                comparison_analysis, query_analysis, subgraph
            )
            
            # 6. 限制token数量
            final_text = ProcessorUtils.limit_tokens(
                comparison_text, 
                self.config.max_tokens
            )
            
            # 7. 构建结果
            result = {
                'success': True,
                'query': query,
                'context': context or {},
                'query_analysis': query_analysis,
                'comparison_entities': list(entity_nodes.keys()),
                'entity_nodes_count': sum(len(nodes) for nodes in entity_nodes.values()),
                'subgraph_summary': {
                    'nodes': subgraph['node_count'],
                    'edges': subgraph['edge_count'],
                    'algorithm': subgraph.get('algorithm', 'pcst')
                },
                'comparison_analysis': comparison_analysis,
                'contextualized_text': final_text,
                'processing_strategy': 'comparative_analysis',
                'confidence': self._calculate_confidence(entity_nodes, comparison_analysis)
            }
            
            logger.info(f"✅ 比较查询处理完成，比较实体数: {len(entity_nodes)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 比较查询处理失败: {str(e)}")
            raise
    
    def _analyze_comparison_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析比较查询"""
        analysis = {
            'query_type': 'comparative_query',
            'comparison_type': self._detect_comparison_type(query),
            'comparison_entities': self._extract_comparison_entities(query),
            'comparison_aspects': self._extract_comparison_aspects(query),
            'comparison_pattern': self._detect_comparison_pattern(query)
        }
        
        # 分析上下文
        if context:
            analysis['context_entities'] = context.get('entities', [])
            analysis['context_comparisons'] = context.get('comparisons', [])
        
        return analysis
    
    def _detect_comparison_type(self, query: str) -> str:
        """检测比较类型"""
        query_lower = query.lower()
        
        # 优劣比较
        if any(word in query for word in ['更好', '更强', '更厉害', '谁更', '哪个更']):
            return 'superiority'
        
        # 相似性比较
        elif any(word in query for word in ['相似', '类似', '差不多', '像']):
            return 'similarity'
        
        # 差异比较
        elif any(word in query for word in ['不同', '区别', '差异', '对比']):
            return 'difference'
        
        # 数量比较
        elif any(word in query for word in ['多少', '哪个多', '哪个少', '更多']):
            return 'quantity'
        
        # 排名比较
        elif any(word in query for word in ['排名', '排行', '第一', '最好']):
            return 'ranking'
        
        return 'general'
    
    def _extract_comparison_entities(self, query: str) -> List[str]:
        """提取比较实体"""
        entities = []
        
        # 简单的实体提取逻辑
        # 查找连接词分隔的实体
        conjunctions = ['和', '与', '跟', '还是', '或者']
        
        for conjunction in conjunctions:
            if conjunction in query:
                parts = query.split(conjunction)
                if len(parts) == 2:
                    # 尝试从每部分提取实体名称
                    entity1 = self._extract_entity_from_text(parts[0])
                    entity2 = self._extract_entity_from_text(parts[1])
                    
                    if entity1:
                        entities.append(entity1)
                    if entity2:
                        entities.append(entity2)
                    break
        
        return entities
    
    def _extract_entity_from_text(self, text: str) -> Optional[str]:
        """从文本中提取实体名称"""
        # 简单的实体提取（可以后续改进）
        text = text.strip()
        
        # 常见球员名称
        player_names = ['科比', '詹姆斯', '乔丹', '库里', '杜兰特', '威少', '哈登']
        for name in player_names:
            if name in text:
                return name
        
        # 常见球队名称
        team_names = ['湖人', '勇士', '公牛', '马刺', '骑士', '热火', '凯尔特人']
        for name in team_names:
            if name in text:
                return name
        
        # 提取可能的实体（中文字符序列）
        import re
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        matches = chinese_pattern.findall(text)
        
        if matches:
            # 返回最长的匹配
            return max(matches, key=len)
        
        return None
    
    def _extract_comparison_aspects(self, query: str) -> List[str]:
        """提取比较方面"""
        aspects = []
        query_lower = query.lower()
        
        # 技能方面
        if any(word in query for word in ['得分', '投篮', '进攻']):
            aspects.append('scoring')
        if any(word in query for word in ['防守', '防守能力']):
            aspects.append('defense')
        if any(word in query for word in ['助攻', '传球']):
            aspects.append('assists')
        if any(word in query for word in ['篮板']):
            aspects.append('rebounds')
        
        # 职业方面
        if any(word in query for word in ['冠军', '总冠军']):
            aspects.append('championships')
        if any(word in query for word in ['MVP', '最有价值球员']):
            aspects.append('mvp')
        if any(word in query for word in ['年龄', '岁数']):
            aspects.append('age')
        
        # 球队方面
        if any(word in query for word in ['战绩', '胜率']):
            aspects.append('record')
        if any(word in query for word in ['球员', '阵容']):
            aspects.append('roster')
        
        return aspects if aspects else ['overall']
    
    def _detect_comparison_pattern(self, query: str) -> Dict[str, Any]:
        """检测比较模式"""
        pattern = {
            'pattern_type': 'binary',  # binary, multiple, ranking
            'comparison_direction': 'neutral',  # positive, negative, neutral
            'expectation': 'objective'  # objective, subjective
        }
        
        # 检测模式类型
        if any(word in query for word in ['排名', '排行', '前几', '最好的']):
            pattern['pattern_type'] = 'ranking'
        elif len(self._extract_comparison_entities(query)) > 2:
            pattern['pattern_type'] = 'multiple'
        
        # 检测比较方向
        if any(word in query for word in ['更好', '更强', '更厉害']):
            pattern['comparison_direction'] = 'positive'
        elif any(word in query for word in ['更差', '不如', '较弱']):
            pattern['comparison_direction'] = 'negative'
        
        # 检测期望类型
        if any(word in query for word in ['我觉得', '我认为', '个人认为']):
            pattern['expectation'] = 'subjective'
        
        return pattern
    
    def _retrieve_comparison_entities(self, query: str, entities: List[str]) -> Dict[str, List[Dict]]:
        """检索比较实体"""
        entity_nodes = {}
        
        if not entities:
            # 如果没有明确实体，使用整个查询检索
            all_nodes = self.retriever.retrieve(query, top_k=10)
            entity_nodes['general'] = all_nodes
        else:
            # 为每个实体单独检索
            for entity in entities:
                entity_query = f"{entity} {query}"
                nodes = self.retriever.retrieve(entity_query, top_k=6)
                if nodes:
                    entity_nodes[entity] = nodes
        
        return entity_nodes
    
    def _perform_comparison_analysis(self, subgraph: Dict[str, Any], 
                                   entity_nodes: Dict[str, List[Dict]], 
                                   query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """执行比较分析"""
        analysis = {
            'entity_profiles': {},
            'comparison_metrics': {},
            'similarity_analysis': {},
            'difference_analysis': {},
            'conclusion': ''
        }
        
        # 构建实体档案
        for entity_name, nodes in entity_nodes.items():
            profile = self._build_entity_profile(entity_name, nodes, subgraph)
            analysis['entity_profiles'][entity_name] = profile
        
        # 计算比较指标
        if len(entity_nodes) >= 2:
            analysis['comparison_metrics'] = self._calculate_comparison_metrics(
                analysis['entity_profiles']
            )
            
            analysis['similarity_analysis'] = self._analyze_similarities(
                analysis['entity_profiles']
            )
            
            analysis['difference_analysis'] = self._analyze_differences(
                analysis['entity_profiles']
            )
            
            analysis['conclusion'] = self._generate_comparison_conclusion(
                analysis, query_analysis
            )
        
        return analysis
    
    def _build_entity_profile(self, entity_name: str, nodes: List[Dict], 
                            subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """构建实体档案"""
        profile = {
            'name': entity_name,
            'type': 'unknown',
            'attributes': {},
            'connections': [],
            'importance_score': 0.0
        }
        
        # 分析节点类型和属性
        entity_nodes = [n for n in nodes if entity_name in n.get('name', '')]
        
        if entity_nodes:
            primary_node = entity_nodes[0]
            profile['type'] = primary_node.get('type', 'unknown')
            profile['attributes'] = primary_node.get('properties', {})
            profile['importance_score'] = primary_node.get('similarity', 0.0)
        
        # 分析连接
        subgraph_edges = subgraph.get('edges', [])
        for edge in subgraph_edges:
            if entity_name in edge.get('source', '') or entity_name in edge.get('target', ''):
                connection = {
                    'relation': edge.get('relation', 'unknown'),
                    'target': edge.get('target', '') if entity_name in edge.get('source', '') else edge.get('source', ''),
                    'weight': edge.get('weight', 1.0)
                }
                profile['connections'].append(connection)
        
        return profile
    
    def _calculate_comparison_metrics(self, entity_profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """计算比较指标"""
        metrics = {
            'attribute_comparison': {},
            'connection_comparison': {},
            'overall_scores': {}
        }
        
        entity_names = list(entity_profiles.keys())
        
        # 属性比较
        for entity_name, profile in entity_profiles.items():
            attributes = profile.get('attributes', {})
            
            # 数值属性比较
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, (int, float)):
                    if attr_name not in metrics['attribute_comparison']:
                        metrics['attribute_comparison'][attr_name] = {}
                    metrics['attribute_comparison'][attr_name][entity_name] = attr_value
        
        # 连接度比较
        for entity_name, profile in entity_profiles.items():
            connection_count = len(profile.get('connections', []))
            metrics['connection_comparison'][entity_name] = connection_count
        
        # 整体分数
        for entity_name, profile in entity_profiles.items():
            overall_score = (
                profile.get('importance_score', 0.0) * 0.4 +
                len(profile.get('connections', [])) * 0.3 +
                len(profile.get('attributes', {})) * 0.3
            )
            metrics['overall_scores'][entity_name] = overall_score
        
        return metrics
    
    def _analyze_similarities(self, entity_profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """分析相似性"""
        similarities = {
            'common_attributes': [],
            'common_connections': [],
            'similarity_score': 0.0
        }
        
        entity_names = list(entity_profiles.keys())
        if len(entity_names) < 2:
            return similarities
        
        # 查找共同属性
        all_attributes = []
        for profile in entity_profiles.values():
            all_attributes.append(set(profile.get('attributes', {}).keys()))
        
        if all_attributes:
            common_attrs = set.intersection(*all_attributes)
            similarities['common_attributes'] = list(common_attrs)
        
        # 查找共同连接
        all_connections = []
        for profile in entity_profiles.values():
            connections = set(conn.get('target', '') for conn in profile.get('connections', []))
            all_connections.append(connections)
        
        if all_connections:
            common_connections = set.intersection(*all_connections)
            similarities['common_connections'] = list(common_connections)
        
        # 计算相似度分数
        similarity_factors = [
            len(similarities['common_attributes']) * 0.4,
            len(similarities['common_connections']) * 0.6
        ]
        similarities['similarity_score'] = sum(similarity_factors) / 10  # 归一化
        
        return similarities
    
    def _analyze_differences(self, entity_profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """分析差异"""
        differences = {
            'unique_attributes': {},
            'unique_connections': {},
            'attribute_differences': {},
            'difference_score': 0.0
        }
        
        entity_names = list(entity_profiles.keys())
        
        # 查找独特属性
        for entity_name, profile in entity_profiles.items():
            attributes = set(profile.get('attributes', {}).keys())
            
            other_attributes = set()
            for other_name, other_profile in entity_profiles.items():
                if other_name != entity_name:
                    other_attributes.update(other_profile.get('attributes', {}).keys())
            
            unique_attrs = attributes - other_attributes
            differences['unique_attributes'][entity_name] = list(unique_attrs)
        
        # 查找独特连接
        for entity_name, profile in entity_profiles.items():
            connections = set(conn.get('target', '') for conn in profile.get('connections', []))
            
            other_connections = set()
            for other_name, other_profile in entity_profiles.items():
                if other_name != entity_name:
                    other_connections.update(
                        conn.get('target', '') for conn in other_profile.get('connections', [])
                    )
            
            unique_connections = connections - other_connections
            differences['unique_connections'][entity_name] = list(unique_connections)
        
        # 属性值差异
        for entity_name, profile in entity_profiles.items():
            attributes = profile.get('attributes', {})
            for attr_name, attr_value in attributes.items():
                if isinstance(attr_value, (int, float)):
                    if attr_name not in differences['attribute_differences']:
                        differences['attribute_differences'][attr_name] = {}
                    differences['attribute_differences'][attr_name][entity_name] = attr_value
        
        # 计算差异分数
        total_unique_attrs = sum(len(attrs) for attrs in differences['unique_attributes'].values())
        total_unique_connections = sum(len(conns) for conns in differences['unique_connections'].values())
        
        differences['difference_score'] = (total_unique_attrs * 0.4 + total_unique_connections * 0.6) / 10
        
        return differences
    
    def _generate_comparison_conclusion(self, analysis: Dict[str, Any], 
                                      query_analysis: Dict[str, Any]) -> str:
        """生成比较结论"""
        comparison_type = query_analysis.get('comparison_type', 'general')
        entity_profiles = analysis.get('entity_profiles', {})
        metrics = analysis.get('comparison_metrics', {})
        
        if len(entity_profiles) < 2:
            return "需要至少两个实体进行比较。"
        
        entity_names = list(entity_profiles.keys())
        
        if comparison_type == 'superiority':
            # 优劣比较
            overall_scores = metrics.get('overall_scores', {})
            if overall_scores:
                best_entity = max(overall_scores, key=overall_scores.get)
                return f"基于综合分析，{best_entity} 在相关指标上表现更好。"
        
        elif comparison_type == 'similarity':
            # 相似性比较
            similarity_analysis = analysis.get('similarity_analysis', {})
            similarity_score = similarity_analysis.get('similarity_score', 0.0)
            
            if similarity_score > 0.5:
                return f"{' 和 '.join(entity_names)} 有较多相似之处。"
            else:
                return f"{' 和 '.join(entity_names)} 差异较大。"
        
        elif comparison_type == 'difference':
            # 差异比较
            difference_analysis = analysis.get('difference_analysis', {})
            difference_score = difference_analysis.get('difference_score', 0.0)
            
            if difference_score > 0.5:
                return f"{' 和 '.join(entity_names)} 在多个方面存在显著差异。"
            else:
                return f"{' 和 '.join(entity_names)} 比较相似。"
        
        return f"已完成 {' 和 '.join(entity_names)} 的比较分析。"
    
    def _generate_comparison_text(self, comparison_analysis: Dict[str, Any], 
                                query_analysis: Dict[str, Any], 
                                subgraph: Dict[str, Any]) -> str:
        """生成比较文本"""
        text_parts = []
        text_parts.append("=== 比较分析结果 ===")
        
        entity_profiles = comparison_analysis.get('entity_profiles', {})
        
        # 实体概述
        if entity_profiles:
            text_parts.append("\n## 比较实体:")
            for entity_name, profile in entity_profiles.items():
                entity_type = profile.get('type', '未知')
                text_parts.append(f"- {entity_name} ({entity_type})")
        
        # 比较指标
        metrics = comparison_analysis.get('comparison_metrics', {})
        if metrics.get('overall_scores'):
            text_parts.append("\n## 综合评分:")
            for entity_name, score in metrics['overall_scores'].items():
                text_parts.append(f"- {entity_name}: {score:.2f}")
        
        # 相似性分析
        similarity_analysis = comparison_analysis.get('similarity_analysis', {})
        if similarity_analysis.get('common_attributes'):
            text_parts.append(f"\n## 共同特征:")
            for attr in similarity_analysis['common_attributes']:
                text_parts.append(f"- {attr}")
        
        # 差异分析
        difference_analysis = comparison_analysis.get('difference_analysis', {})
        if difference_analysis.get('unique_attributes'):
            text_parts.append(f"\n## 独特特征:")
            for entity_name, unique_attrs in difference_analysis['unique_attributes'].items():
                if unique_attrs:
                    text_parts.append(f"- {entity_name}: {', '.join(unique_attrs)}")
        
        # 结论
        conclusion = comparison_analysis.get('conclusion', '')
        if conclusion:
            text_parts.append(f"\n## 结论:")
            text_parts.append(conclusion)
        
        return "\n".join(text_parts)
    
    def _create_empty_result(self, query: str, reason: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'comparative_query'},
            'comparison_entities': [],
            'entity_nodes_count': 0,
            'subgraph_summary': {'nodes': 0, 'edges': 0, 'algorithm': 'none'},
            'comparison_analysis': {'entity_profiles': {}, 'conclusion': ''},
            'contextualized_text': f"Sorry, {reason}. Please provide specific comparison objects.",
            'processing_strategy': 'comparative_analysis',
            'confidence': 0.0,
            'empty_reason': reason
        }
    
    def _create_fallback_result(self, query: str, entity_nodes: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """创建回退结果"""
        fallback_text = self._format_comparison_fallback(entity_nodes, query)
        
        return {
            'success': True,
            'query': query,
            'context': {},
            'query_analysis': {'query_type': 'comparative_query'},
            'comparison_entities': list(entity_nodes.keys()),
            'entity_nodes_count': sum(len(nodes) for nodes in entity_nodes.values()),
            'subgraph_summary': {'nodes': 0, 'edges': 0, 'algorithm': 'fallback'},
            'comparison_analysis': {'entity_profiles': {}, 'conclusion': ''},
            'contextualized_text': fallback_text,
            'processing_strategy': 'comparative_analysis_fallback',
            'confidence': 0.6
        }
    
    def _format_comparison_fallback(self, entity_nodes: Dict[str, List[Dict]], query: str) -> str:
        """格式化比较回退结果"""
        text_parts = ["基于检索结果的简单比较："]
        
        for entity_name, nodes in entity_nodes.items():
            if nodes:
                text_parts.append(f"\n{entity_name}:")
                for node in nodes[:3]:  # 最多显示3个节点
                    name = node.get('name', '未知')
                    node_type = node.get('type', '未知')
                    text_parts.append(f"  - {name} ({node_type})")
        
        text_parts.append("\n注：需要更完整的图结构进行详细比较。")
        
        return "\n".join(text_parts)
    
    def _calculate_confidence(self, entity_nodes: Dict[str, List[Dict]], 
                            comparison_analysis: Dict[str, Any]) -> float:
        """计算结果置信度"""
        if not entity_nodes:
            return 0.0
        
        # 基础置信度
        base_confidence = 0.55
        
        # 实体数量加成
        entity_count = len(entity_nodes)
        if entity_count >= 2:
            entity_bonus = min(0.2, entity_count * 0.1)
        else:
            entity_bonus = -0.2  # 缺少比较对象
        
        # 检索质量加成
        all_similarities = []
        for nodes in entity_nodes.values():
            similarities = [node.get('similarity', 0) for node in nodes]
            all_similarities.extend(similarities)
        
        if all_similarities:
            avg_similarity = sum(all_similarities) / len(all_similarities)
            similarity_bonus = avg_similarity * 0.15
        else:
            similarity_bonus = 0
        
        # 比较分析完整性加成
        analysis_bonus = 0
        if comparison_analysis.get('entity_profiles'):
            analysis_bonus += 0.1
        if comparison_analysis.get('conclusion'):
            analysis_bonus += 0.1
        
        total_confidence = base_confidence + entity_bonus + similarity_bonus + analysis_bonus
        return min(1.0, max(0.0, total_confidence))

# =============================================================================
# 工厂函数
# =============================================================================

def create_comparison_processor(custom_config: Optional[Dict[str, Any]] = None) -> ComparisonProcessor:
    """创建比较查询处理器实例"""
    if custom_config:
        config = ProcessorDefaultConfigs.get_comparison_processor_config()
        
        # 更新配置
        if 'retriever' in custom_config:
            config.retriever_config.config.update(custom_config['retriever'])
        if 'graph_builder' in custom_config:
            config.graph_builder_config.config.update(custom_config['graph_builder'])
        if 'textualizer' in custom_config:
            config.textualizer_config.config.update(custom_config['textualizer'])
        
        for key in ['cache_enabled', 'cache_ttl', 'max_tokens']:
            if key in custom_config:
                setattr(config, key, custom_config[key])
        
        return ComparisonProcessor(config)
    else:
        return ComparisonProcessor()
