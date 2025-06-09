"""
G-Retriever 图构造器
实现基于 PCST (Prize-Collecting Steiner Tree) 算法的智能子图构建
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from app.database import NebulaGraphConnection
import logging

logger = logging.getLogger(__name__)

class GraphConstructor:
    """智能子图构造器"""
    
    def __init__(self):
        """初始化图构造器"""
        self.nebula_conn = NebulaGraphConnection()
        self.graph = nx.Graph()
        self.node_prizes = {}  # 节点奖励值
        self.edge_costs = {}   # 边成本值
        
    def initialize(self) -> bool:
        """初始化构造器"""
        try:
            logger.info("🔄 正在初始化图构造器...")
            
            if not self.nebula_conn.connect():
                logger.error("❌ NebulaGraph连接失败")
                return False
            
            # 构建完整图结构
            self._build_full_graph()
            
            logger.info("✅ 图构造器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 图构造器初始化失败: {str(e)}")
            return False
    
    def _build_full_graph(self):
        """构建完整的图结构"""
        logger.info("🏗️  正在构建完整图结构...")
        
        # 添加球员节点
        self._add_player_nodes()
        
        # 添加球队节点
        self._add_team_nodes()
        
        # 添加关系边
        self._add_serve_edges()
        
        logger.info(f"✅ 图结构构建完成，节点数: {self.graph.number_of_nodes()}, 边数: {self.graph.number_of_edges()}")
    
    def _add_player_nodes(self):
        """添加球员节点"""
        query = """
        MATCH (p:player)
        RETURN p.player.name AS name, p.player.age AS age
        LIMIT 100
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                name, age = row[0], row[1]
                node_id = f"player:{name}"
                
                # 添加节点到图
                self.graph.add_node(node_id, 
                                  type='player', 
                                  name=name, 
                                  age=age,
                                  prize=1.0)  # 默认奖励值
                
                # 设置节点奖励
                self.node_prizes[node_id] = 1.0
    
    def _add_team_nodes(self):
        """添加球队节点"""
        query = """
        MATCH (t:team)
        RETURN t.team.name AS name
        LIMIT 50
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                team_name = row[0]
                node_id = f"team:{team_name}"
                
                # 添加节点到图
                self.graph.add_node(node_id, 
                                  type='team', 
                                  name=team_name,
                                  prize=1.0)  # 默认奖励值
                
                # 设置节点奖励
                self.node_prizes[node_id] = 1.0
    
    def _add_serve_edges(self):
        """添加效力关系边"""
        query = """
        MATCH (p:player)-[s:serve]->(t:team)
        RETURN p.player.name AS player, t.team.name AS team
        LIMIT 200
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                player_name, team_name = row[0], row[1]
                source = f"player:{player_name}"
                target = f"team:{team_name}"
                
                # 只添加存在的节点之间的边
                if self.graph.has_node(source) and self.graph.has_node(target):
                    self.graph.add_edge(source, target, 
                                      type='serve',
                                      cost=1.0)  # 默认成本值
                    
                    # 设置边成本
                    edge_key = (source, target)
                    self.edge_costs[edge_key] = 1.0
    
    def set_node_prizes(self, node_similarities: List[Dict[str, Any]]):
        """基于相似度设置节点奖励值"""
        # 调试信息
        if node_similarities:
            logger.debug(f"设置节点奖励，收到数据样例: {node_similarities[0]}")
        
        # 重置所有节点奖励为较低值
        for node_id in self.graph.nodes():
            self.node_prizes[node_id] = 0.1
            self.graph.nodes[node_id]['prize'] = 0.1
        
        # 为相似节点设置高奖励值
        for node_data in node_similarities:
            # 兼容不同的键名
            node_id = node_data.get('node_id') or node_data.get('id')
            similarity = node_data.get('similarity', 0.0)
            
            if node_id and node_id in self.graph.nodes():
                # 奖励值与相似度成正比
                prize = max(similarity * 10, 0.1)  # 确保最小奖励值
                self.node_prizes[node_id] = prize
                self.graph.nodes[node_id]['prize'] = prize
    
    def set_edge_costs(self, edge_similarities: List[Dict[str, Any]]):
        """基于相似度设置边成本值"""
        # 调试信息
        if edge_similarities:
            logger.debug(f"设置边成本，收到数据样例: {edge_similarities[0]}")
        
        # 重置所有边成本为较高值
        for edge in self.graph.edges():
            self.edge_costs[edge] = 5.0
            self.graph.edges[edge]['cost'] = 5.0
        
        # 为相似边设置低成本值
        for edge_data in edge_similarities:
            # 兼容不同的键名
            edge_id = edge_data.get('edge_id') or edge_data.get('id')
            source = edge_data.get('source')
            target = edge_data.get('target')
            similarity = edge_data.get('similarity', 0.0)
            
            # 检查边是否存在
            if source and target and self.graph.has_edge(source, target):
                # 成本值与相似度成反比
                cost = max(1.0 - similarity, 0.1)  # 高相似度 = 低成本
                self.edge_costs[(source, target)] = cost
                self.graph.edges[source, target]['cost'] = cost
    
    def pcst_subgraph(self, seed_nodes: List[str], max_nodes: int = 20) -> nx.Graph:
        """使用 PCST 算法构建子图"""
        logger.info(f"🌱 开始构建子图，种子节点: {seed_nodes}")
        
        if not seed_nodes:
            return nx.Graph()
        
        # 简化版的 PCST 算法实现
        # 1. 从种子节点开始
        subgraph_nodes = set(seed_nodes)
        
        # 2. 贪心扩展：添加高奖励/低成本的邻居节点
        current_nodes = set(seed_nodes)
        
        while len(subgraph_nodes) < max_nodes:
            best_node = None
            best_score = -float('inf')
            
            # 检查当前节点的所有邻居
            candidates = set()
            for node in current_nodes:
                if node in self.graph:
                    candidates.update(self.graph.neighbors(node))
            
            # 移除已经在子图中的节点
            candidates -= subgraph_nodes
            
            if not candidates:
                break
            
            # 选择最佳候选节点
            for candidate in candidates:
                if candidate in self.node_prizes:
                    prize = self.node_prizes[candidate]
                    
                    # 计算连接成本（到已选择节点的最小成本）
                    min_cost = float('inf')
                    for selected_node in subgraph_nodes:
                        if self.graph.has_edge(candidate, selected_node):
                            edge_key = (candidate, selected_node)
                            cost = self.edge_costs.get(edge_key, 1.0)
                            min_cost = min(min_cost, cost)
                    
                    if min_cost == float('inf'):
                        continue
                    
                    # 计算收益分数 (奖励 - 成本)
                    score = prize - min_cost
                    
                    if score > best_score:
                        best_score = score
                        best_node = candidate
            
            # 如果找到好的节点且分数为正，添加它
            if best_node and best_score > 0:
                subgraph_nodes.add(best_node)
                current_nodes.add(best_node)
            else:
                break
        
        # 3. 构建包含连接边的子图
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        # 4. 确保子图连通性（添加最短路径）
        subgraph = self._ensure_connectivity(subgraph, seed_nodes)
        
        logger.info(f"✅ 子图构建完成，节点数: {subgraph.number_of_nodes()}, 边数: {subgraph.number_of_edges()}")
        return subgraph
    
    def _ensure_connectivity(self, subgraph: nx.Graph, seed_nodes: List[str]) -> nx.Graph:
        """确保子图连通性"""
        if subgraph.number_of_nodes() <= 1:
            return subgraph
        
        # 检查连通性
        if nx.is_connected(subgraph):
            return subgraph
        
        # 如果不连通，添加最短路径来连接组件
        components = list(nx.connected_components(subgraph))
        
        if len(components) <= 1:
            return subgraph
        
        # 连接最大组件和其他组件
        largest_component = max(components, key=len)
        
        for component in components:
            if component == largest_component:
                continue
            
            # 找到连接两个组件的最短路径
            min_path_length = float('inf')
            best_path = None
            
            for node1 in largest_component:
                for node2 in component:
                    if node1 in self.graph and node2 in self.graph:
                        try:
                            path = nx.shortest_path(self.graph, node1, node2)
                            if len(path) < min_path_length:
                                min_path_length = len(path)
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue
            
            # 添加最短路径上的节点和边
            if best_path:
                for node in best_path:
                    subgraph.add_node(node, **self.graph.nodes[node])
                
                for i in range(len(best_path) - 1):
                    node1, node2 = best_path[i], best_path[i + 1]
                    if self.graph.has_edge(node1, node2):
                        subgraph.add_edge(node1, node2, **self.graph.edges[node1, node2])
        
        return subgraph
    
    def extract_subgraph_info(self, subgraph: nx.Graph) -> Dict[str, Any]:
        """提取子图信息"""
        nodes_info = []
        edges_info = []
        
        # 提取节点信息
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            nodes_info.append({
                'id': node_id,
                'type': node_data.get('type', 'unknown'),
                'name': node_data.get('name', ''),
                'properties': {k: v for k, v in node_data.items() 
                             if k not in ['type', 'name', 'prize']}
            })
        
        # 提取边信息
        for source, target in subgraph.edges():
            edge_data = subgraph.edges[source, target]
            edges_info.append({
                'source': source,
                'target': target,
                'type': edge_data.get('type', 'unknown'),
                'properties': {k: v for k, v in edge_data.items() 
                             if k not in ['type', 'cost']}
            })
        
        return {
            'nodes': nodes_info,
            'edges': edges_info,
            'num_nodes': len(nodes_info),
            'num_edges': len(edges_info)
        }
    
    def close(self):
        """关闭连接"""
        if self.nebula_conn:
            self.nebula_conn.close()
