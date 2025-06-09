"""
G-Retriever 上下文格式化器
将子图结构转换为自然语言格式，用于LLM理解
"""
import networkx as nx
from typing import List, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

class ContextFormatter:
    """图到文本转换器"""
    
    def __init__(self):
        """初始化格式化器"""
        self.entity_descriptions = {}  # 实体描述缓存
        
    def format_subgraph_to_text(self, subgraph_info: Dict[str, Any], query: str = "") -> str:
        """将子图信息转换为结构化文本"""
        nodes = subgraph_info.get('nodes', [])
        edges = subgraph_info.get('edges', [])
        
        if not nodes:
            return "没有找到相关的图结构信息。"
        
        # 构建格式化文本
        text_parts = []
        
        # 1. 添加概述
        text_parts.append(f"**图结构概述**")
        text_parts.append(f"- 节点数量: {len(nodes)}")
        text_parts.append(f"- 边数量: {len(edges)}")
        text_parts.append("")
        
        # 2. 添加实体信息
        text_parts.append("**实体信息**")
        
        # 按类型分组节点
        nodes_by_type = {}
        for node in nodes:
            node_type = node.get('type', 'unknown')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        # 格式化每种类型的节点
        for node_type, type_nodes in nodes_by_type.items():
            if node_type == 'player':
                text_parts.append("*球员:*")
                for node in type_nodes:
                    name = node.get('name', '')
                    props = node.get('properties', {})
                    age = props.get('age', '未知')
                    text_parts.append(f"  - {name} (年龄: {age}岁)")
                text_parts.append("")
                
            elif node_type == 'team':
                text_parts.append("*球队:*")
                for node in type_nodes:
                    name = node.get('name', '')
                    text_parts.append(f"  - {name}")
                text_parts.append("")
        
        # 3. 添加关系信息
        if edges:
            text_parts.append("**关系信息**")
            
            # 按关系类型分组
            edges_by_type = {}
            for edge in edges:
                edge_type = edge.get('type', 'unknown')
                if edge_type not in edges_by_type:
                    edges_by_type[edge_type] = []
                edges_by_type[edge_type].append(edge)
            
            # 格式化每种类型的关系
            for edge_type, type_edges in edges_by_type.items():
                if edge_type == 'serve':
                    text_parts.append("*效力关系:*")
                    for edge in type_edges:
                        source_name = self._extract_name_from_id(edge.get('source', ''))
                        target_name = self._extract_name_from_id(edge.get('target', ''))
                        text_parts.append(f"  - {source_name} 效力于 {target_name}")
                    text_parts.append("")
        
        # 4. 添加关键路径分析
        key_paths = self._extract_key_paths(nodes, edges, query)
        if key_paths:
            text_parts.append("**关键路径**")
            for i, path in enumerate(key_paths[:3], 1):  # 最多显示3条路径
                text_parts.append(f"{i}. {path}")
            text_parts.append("")
        
        return "\n".join(text_parts)
    
    def format_subgraph_to_qa_context(self, subgraph_info: Dict[str, Any], query: str) -> str:
        """为问答系统格式化上下文"""
        nodes = subgraph_info.get('nodes', [])
        edges = subgraph_info.get('edges', [])
        
        context_parts = []
        
        # 1. 相关实体
        context_parts.append("相关实体:")
        
        players = [n for n in nodes if n.get('type') == 'player']
        teams = [n for n in nodes if n.get('type') == 'team']
        
        if players:
            player_names = [p.get('name', '') for p in players]
            context_parts.append(f"球员: {', '.join(player_names)}")
        
        if teams:
            team_names = [t.get('name', '') for t in teams]
            context_parts.append(f"球队: {', '.join(team_names)}")
        
        context_parts.append("")
        
        # 2. 关键事实
        context_parts.append("关键事实:")
        
        # 添加球员详细信息
        for player in players:
            name = player.get('name', '')
            props = player.get('properties', {})
            age = props.get('age')
            if age:
                context_parts.append(f"- {name}的年龄是{age}岁")
        
        # 添加效力关系
        serve_edges = [e for e in edges if e.get('type') == 'serve']
        for edge in serve_edges:
            player_name = self._extract_name_from_id(edge.get('source', ''))
            team_name = self._extract_name_from_id(edge.get('target', ''))
            context_parts.append(f"- {player_name}效力于{team_name}")
        
        # 3. 添加推理链
        reasoning_chain = self._build_reasoning_chain(nodes, edges, query)
        if reasoning_chain:
            context_parts.append("")
            context_parts.append("推理链:")
            context_parts.append(reasoning_chain)
        
        return "\n".join(context_parts)
    
    def format_subgraph_compact(self, subgraph_info: Dict[str, Any]) -> str:
        """紧凑格式化子图信息"""
        nodes = subgraph_info.get('nodes', [])
        edges = subgraph_info.get('edges', [])
        
        facts = []
        
        # 实体事实
        for node in nodes:
            node_type = node.get('type', '')
            name = node.get('name', '')
            props = node.get('properties', {})
            
            if node_type == 'player' and name:
                age = props.get('age')
                fact = f"{name}是球员"
                if age:
                    fact += f"，年龄{age}岁"
                facts.append(fact)
            elif node_type == 'team' and name:
                facts.append(f"{name}是篮球队")
        
        # 关系事实
        for edge in edges:
            edge_type = edge.get('type', '')
            if edge_type == 'serve':
                player_name = self._extract_name_from_id(edge.get('source', ''))
                team_name = self._extract_name_from_id(edge.get('target', ''))
                if player_name and team_name:
                    facts.append(f"{player_name}效力于{team_name}")
        
        return "; ".join(facts) if facts else "没有相关信息"
    
    def _extract_name_from_id(self, node_id: str) -> str:
        """从节点ID提取名称"""
        if ':' in node_id:
            return node_id.split(':', 1)[1]
        return node_id
    
    def _extract_key_paths(self, nodes: List[Dict], edges: List[Dict], query: str) -> List[str]:
        """提取关键路径"""
        paths = []
        
        # 构建简单的图结构用于路径查找
        graph_dict = {}
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            edge_type = edge.get('type', '')
            
            if source not in graph_dict:
                graph_dict[source] = []
            graph_dict[source].append((target, edge_type))
        
        # 查找涉及多个实体的路径
        for node in nodes:
            node_id = node.get('id', '')
            if node_id in graph_dict:
                for target, edge_type in graph_dict[node_id]:
                    source_name = self._extract_name_from_id(node_id)
                    target_name = self._extract_name_from_id(target)
                    
                    if edge_type == 'serve':
                        paths.append(f"{source_name} → 效力于 → {target_name}")
        
        return paths[:5]  # 返回前5个路径
    
    def _build_reasoning_chain(self, nodes: List[Dict], edges: List[Dict], query: str) -> str:
        """构建推理链"""
        # 简单的推理链构建
        query_lower = query.lower()
        
        # 识别查询中的关键实体
        mentioned_entities = []
        for node in nodes:
            name = node.get('name', '').lower()
            if name and name in query_lower:
                mentioned_entities.append(node)
        
        if not mentioned_entities:
            return ""
        
        reasoning_steps = []
        
        # 添加实体信息
        for entity in mentioned_entities:
            entity_type = entity.get('type', '')
            name = entity.get('name', '')
            
            if entity_type == 'player':
                props = entity.get('properties', {})
                age = props.get('age')
                step = f"1. {name}是一名篮球球员"
                if age:
                    step += f"，年龄{age}岁"
                reasoning_steps.append(step)
        
        # 添加关系推理
        step_num = len(reasoning_steps) + 1
        for edge in edges:
            edge_type = edge.get('type', '')
            if edge_type == 'serve':
                player_name = self._extract_name_from_id(edge.get('source', ''))
                team_name = self._extract_name_from_id(edge.get('target', ''))
                
                # 检查是否涉及查询中的实体
                if (player_name.lower() in query_lower or 
                    team_name.lower() in query_lower):
                    reasoning_steps.append(f"{step_num}. {player_name}效力于{team_name}")
                    step_num += 1
        
        return "\n".join(reasoning_steps)
    
    def format_node_description(self, node: Dict[str, Any]) -> str:
        """格式化单个节点描述"""
        node_type = node.get('type', '')
        name = node.get('name', '')
        props = node.get('properties', {})
        
        if node_type == 'player':
            age = props.get('age', '')
            desc = f"球员{name}"
            if age:
                desc += f"（{age}岁）"
            return desc
        elif node_type == 'team':
            return f"篮球队{name}"
        else:
            return f"{node_type}: {name}"
    
    def format_edge_description(self, edge: Dict[str, Any]) -> str:
        """格式化单个边描述"""
        edge_type = edge.get('type', '')
        source_name = self._extract_name_from_id(edge.get('source', ''))
        target_name = self._extract_name_from_id(edge.get('target', ''))
        
        if edge_type == 'serve':
            return f"{source_name}效力于{target_name}"
        else:
            return f"{source_name} -{edge_type}-> {target_name}"
