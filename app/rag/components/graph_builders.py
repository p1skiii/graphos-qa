"""
RAG 图构建器组件集合
实现多种图构建策略：PCST算法、简单扩展、加权构建
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from app.database import NebulaGraphConnection
import logging
from app.rag.component_factory import BaseGraphBuilder, component_factory

logger = logging.getLogger(__name__)

# =============================================================================
# PCST图构建器
# =============================================================================

class PCSTGraphBuilder(BaseGraphBuilder):
    """PCST (Prize-Collecting Steiner Tree) 图构建器"""
    
    def __init__(self, max_nodes: int = 20, prize_weight: float = 1.0, 
                 cost_weight: float = 0.5, nebula_conn=None):
        """初始化PCST图构建器"""
        self.max_nodes = max_nodes
        self.prize_weight = prize_weight
        self.cost_weight = cost_weight
        
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.graph = nx.Graph()
        self.node_prizes = {}
        self.edge_costs = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化图构建器"""
        try:
            logger.info("🔄 初始化PCST图构建器...")
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 构建完整图结构
            self._build_full_graph()
            
            # 计算节点奖励和边成本
            self._calculate_prizes_and_costs()
            
            self.is_initialized = True
            logger.info("✅ PCST图构建器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ PCST图构建器初始化失败: {str(e)}")
            return False
    
    def _build_full_graph(self):
        """构建完整的图结构"""
        logger.info("🏗️ 构建完整图结构...")
        
        # 添加球员节点
        self._add_player_nodes()
        
        # 添加球队节点
        self._add_team_nodes()
        
        # 添加球员-球队关系
        self._add_player_team_edges()
        
        logger.info(f"✅ 图结构构建完成，节点: {len(self.graph.nodes)}, 边: {len(self.graph.edges)}")
    
    def _add_player_nodes(self):
        """添加球员节点"""
        query = """
        MATCH (p:player)
        RETURN p.player.name AS name, p.player.age AS age
        LIMIT 200
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                name, age = row[0], row[1]
                # 处理NULL值
                if name is None:
                    continue
                if age is None:
                    age = 0
                    
                node_id = f"player:{name}"
                
                self.graph.add_node(node_id, 
                                  type='player', 
                                  name=name, 
                                  age=age,
                                  description=f"球员 {name}, 年龄 {age} 岁")
    
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
                name = row[0]
                # 处理NULL值
                if name is None:
                    continue
                    
                node_id = f"team:{name}"
                
                self.graph.add_node(node_id, 
                                  type='team', 
                                  name=name,
                                  description=f"球队 {name}")
    
    def _add_player_team_edges(self):
        """添加球员-球队关系边"""
        query = """
        MATCH (p:player)-[r:serve]->(t:team)
        RETURN p.player.name AS player_name, t.team.name AS team_name
        LIMIT 500
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                player_name, team_name = row[0], row[1]
                # 处理NULL值
                if player_name is None or team_name is None:
                    continue
                    
                player_id = f"player:{player_name}"
                team_id = f"team:{team_name}"
                
                if player_id in self.graph.nodes and team_id in self.graph.nodes:
                    self.graph.add_edge(player_id, team_id, 
                                      relation='serve',
                                      weight=1.0)
    
    def _calculate_prizes_and_costs(self):
        """计算节点奖励和边成本"""
        logger.info("💰 计算节点奖励和边成本...")
        
        # 计算节点奖励（基于度数和类型）
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            degree = self.graph.degree(node_id)
            
            if node_data['type'] == 'player':
                base_prize = 1.0
            else:  # team
                base_prize = 0.8
            
            # 度数越高，奖励越高
            prize = base_prize + (degree * 0.1)
            self.node_prizes[node_id] = prize
        
        # 计算边成本（基于权重）
        for edge in self.graph.edges:
            edge_data = self.graph.edges[edge]
            cost = edge_data.get('weight', 1.0)
            self.edge_costs[edge] = cost
    
    def build_subgraph(self, seed_nodes: List[str], query: str) -> Dict[str, Any]:
        """使用PCST算法构建子图"""
        if not self.is_initialized:
            raise RuntimeError("图构建器未初始化")
        
        logger.info(f"🌱 使用PCST算法构建子图，种子节点: {len(seed_nodes)}")
        
        # 验证种子节点
        valid_seeds = [node for node in seed_nodes if node in self.graph.nodes]
        if not valid_seeds:
            logger.warning("⚠️ 没有有效的种子节点")
            return self._create_empty_subgraph()
        
        # 运行PCST算法
        try:
            subgraph_nodes = self._run_pcst_algorithm(valid_seeds, query)
            subgraph = self._extract_subgraph(subgraph_nodes)
            
            logger.info(f"✅ PCST子图构建完成，节点: {len(subgraph['nodes'])}, 边: {len(subgraph['edges'])}")
            return subgraph
            
        except Exception as e:
            logger.error(f"❌ PCST算法执行失败: {str(e)}")
            return self._create_fallback_subgraph(valid_seeds)
    
    def _run_pcst_algorithm(self, seed_nodes: List[str], query: str) -> Set[str]:
        """运行PCST算法"""
        # 为种子节点设置更高的奖励
        adjusted_prizes = self.node_prizes.copy()
        for seed in seed_nodes:
            if seed in adjusted_prizes:
                adjusted_prizes[seed] *= 3.0  # 种子节点奖励提升3倍
        
        # 简化的PCST算法实现
        selected_nodes = set(seed_nodes)
        candidates = set()
        
        # 获取种子节点的邻居作为候选
        for seed in seed_nodes:
            if seed in self.graph.nodes:
                candidates.update(self.graph.neighbors(seed))
        
        # 贪心选择节点
        while len(selected_nodes) < self.max_nodes and candidates:
            best_node = None
            best_score = -float('inf')
            
            for candidate in candidates:
                if candidate in selected_nodes:
                    continue
                
                # 计算收益
                prize = adjusted_prizes.get(candidate, 0)
                
                # 计算连接成本
                connection_cost = 0
                for selected in selected_nodes:
                    if self.graph.has_edge(candidate, selected):
                        connection_cost += self.edge_costs.get((candidate, selected), 1.0)
                
                # 计算净收益
                score = prize * self.prize_weight - connection_cost * self.cost_weight
                
                if score > best_score:
                    best_score = score
                    best_node = candidate
            
            if best_node and best_score > 0:
                selected_nodes.add(best_node)
                # 添加新候选节点
                candidates.update(self.graph.neighbors(best_node))
            else:
                break
        
        return selected_nodes
    
    def _extract_subgraph(self, nodes: Set[str]) -> Dict[str, Any]:
        """从完整图中提取子图"""
        subgraph = self.graph.subgraph(nodes)
        
        # 转换为标准格式
        nodes_data = []
        edges_data = []
        
        for node_id in subgraph.nodes:
            node_data = subgraph.nodes[node_id].copy()
            node_data['id'] = node_id
            nodes_data.append(node_data)
        
        for edge in subgraph.edges:
            edge_data = subgraph.edges[edge].copy()
            edge_data['source'] = edge[0]
            edge_data['target'] = edge[1]
            edges_data.append(edge_data)
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'node_count': len(nodes_data),
            'edge_count': len(edges_data),
            'algorithm': 'pcst'
        }
    
    def _create_empty_subgraph(self) -> Dict[str, Any]:
        """创建空子图"""
        return {
            'nodes': [],
            'edges': [],
            'node_count': 0,
            'edge_count': 0,
            'algorithm': 'pcst'
        }
    
    def _create_fallback_subgraph(self, seed_nodes: List[str]) -> Dict[str, Any]:
        """创建回退子图（仅包含种子节点）"""
        nodes_data = []
        for node_id in seed_nodes:
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id].copy()
                node_data['id'] = node_id
                nodes_data.append(node_data)
        
        return {
            'nodes': nodes_data,
            'edges': [],
            'node_count': len(nodes_data),
            'edge_count': 0,
            'algorithm': 'pcst_fallback'
        }

# =============================================================================
# 简单图构建器
# =============================================================================

class SimpleGraphBuilder(BaseGraphBuilder):
    """简单图构建器 - 基于BFS的简单扩展"""
    
    def __init__(self, max_nodes: int = 15, max_depth: int = 2, nebula_conn=None):
        """初始化简单图构建器"""
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.graph = nx.Graph()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化图构建器"""
        try:
            logger.info("🔄 初始化简单图构建器...")
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 构建完整图结构
            self._build_full_graph()
            
            self.is_initialized = True
            logger.info("✅ 简单图构建器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 简单图构建器初始化失败: {str(e)}")
            return False
    
    def _build_full_graph(self):
        """构建完整图结构"""
        logger.info("🏗️ 构建简单图结构...")
        
        # 添加节点和边（简化版本）
        self._add_nodes_and_edges()
        
        logger.info(f"✅ 简单图结构构建完成，节点: {len(self.graph.nodes)}, 边: {len(self.graph.edges)}")
    
    def _add_nodes_and_edges(self):
        """添加节点和边"""
        # 获取球员-球队关系
        query = """
        MATCH (p:player)-[r:serve]->(t:team)
        RETURN p.player.name AS player_name, p.player.age AS player_age,
               t.team.name AS team_name
        LIMIT 300
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                player_name, player_age, team_name = row[0], row[1], row[2]
                
                # 处理NULL值
                if player_name is None or team_name is None:
                    continue
                if player_age is None:
                    player_age = 0
                
                player_id = f"player:{player_name}"
                team_id = f"team:{team_name}"
                
                # 添加球员节点
                if not self.graph.has_node(player_id):
                    self.graph.add_node(player_id,
                                      type='player',
                                      name=player_name,
                                      age=player_age,
                                      description=f"球员 {player_name}, 年龄 {player_age} 岁")
                
                # 添加球队节点
                if not self.graph.has_node(team_id):
                    self.graph.add_node(team_id,
                                      type='team',
                                      name=team_name,
                                      description=f"球队 {team_name}")
                
                # 添加边
                self.graph.add_edge(player_id, team_id,
                                  relation='serve',
                                  weight=1.0)
    
    def build_subgraph(self, seed_nodes: List[str], query: str) -> Dict[str, Any]:
        """使用BFS构建子图"""
        if not self.is_initialized:
            raise RuntimeError("图构建器未初始化")
        
        logger.info(f"🌱 使用BFS构建子图，种子节点: {len(seed_nodes)}")
        
        # 验证种子节点
        valid_seeds = [node for node in seed_nodes if node in self.graph.nodes]
        if not valid_seeds:
            logger.warning("⚠️ 没有有效的种子节点")
            return self._create_empty_subgraph()
        
        # BFS扩展
        selected_nodes = set(valid_seeds)
        current_level = set(valid_seeds)
        
        for depth in range(self.max_depth):
            if len(selected_nodes) >= self.max_nodes:
                break
            
            next_level = set()
            for node in current_level:
                neighbors = list(self.graph.neighbors(node))
                
                # 限制每层添加的节点数
                remaining_slots = self.max_nodes - len(selected_nodes)
                neighbors = neighbors[:remaining_slots]
                
                for neighbor in neighbors:
                    if neighbor not in selected_nodes:
                        next_level.add(neighbor)
                        selected_nodes.add(neighbor)
                        
                        if len(selected_nodes) >= self.max_nodes:
                            break
                
                if len(selected_nodes) >= self.max_nodes:
                    break
            
            current_level = next_level
            if not current_level:
                break
        
        # 提取子图
        subgraph = self._extract_subgraph(selected_nodes)
        
        logger.info(f"✅ BFS子图构建完成，节点: {len(subgraph['nodes'])}, 边: {len(subgraph['edges'])}")
        return subgraph
    
    def _extract_subgraph(self, nodes: Set[str]) -> Dict[str, Any]:
        """从完整图中提取子图"""
        subgraph = self.graph.subgraph(nodes)
        
        nodes_data = []
        edges_data = []
        
        for node_id in subgraph.nodes:
            node_data = subgraph.nodes[node_id].copy()
            node_data['id'] = node_id
            nodes_data.append(node_data)
        
        for edge in subgraph.edges:
            edge_data = subgraph.edges[edge].copy()
            edge_data['source'] = edge[0]
            edge_data['target'] = edge[1]
            edges_data.append(edge_data)
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'node_count': len(nodes_data),
            'edge_count': len(edges_data),
            'algorithm': 'bfs'
        }
    
    def _create_empty_subgraph(self) -> Dict[str, Any]:
        """创建空子图"""
        return {
            'nodes': [],
            'edges': [],
            'node_count': 0,
            'edge_count': 0,
            'algorithm': 'bfs'
        }

# =============================================================================
# 加权图构建器
# =============================================================================

class WeightedGraphBuilder(BaseGraphBuilder):
    """加权图构建器 - 基于节点重要性的图构建"""
    
    def __init__(self, max_nodes: int = 18, importance_threshold: float = 0.1, nebula_conn=None):
        """初始化加权图构建器"""
        self.max_nodes = max_nodes
        self.importance_threshold = importance_threshold
        
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.graph = nx.Graph()
        self.node_importance = {}
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化图构建器"""
        try:
            logger.info("🔄 初始化加权图构建器...")
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 构建图结构
            self._build_full_graph()
            
            # 计算节点重要性
            self._calculate_node_importance()
            
            self.is_initialized = True
            logger.info("✅ 加权图构建器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加权图构建器初始化失败: {str(e)}")
            return False
    
    def _build_full_graph(self):
        """构建完整图结构"""
        logger.info("🏗️ 构建加权图结构...")
        
        # 获取数据并构建图
        query = """
        MATCH (p:player)-[r:serve]->(t:team)
        RETURN p.player.name AS player_name, p.player.age AS player_age,
               t.team.name AS team_name
        LIMIT 300
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                player_name, player_age, team_name = row[0], row[1], row[2]
                
                # 处理NULL值
                if player_name is None or team_name is None:
                    continue
                if player_age is None:
                    player_age = 0
                
                player_id = f"player:{player_name}"
                team_id = f"team:{team_name}"
                
                # 添加节点
                if not self.graph.has_node(player_id):
                    self.graph.add_node(player_id,
                                      type='player',
                                      name=player_name,
                                      age=player_age,
                                      description=f"球员 {player_name}")
                
                if not self.graph.has_node(team_id):
                    self.graph.add_node(team_id,
                                      type='team',
                                      name=team_name,
                                      description=f"球队 {team_name}")
                
                # 添加边
                self.graph.add_edge(player_id, team_id, relation='serve')
        
        logger.info(f"✅ 加权图结构构建完成，节点: {len(self.graph.nodes)}, 边: {len(self.graph.edges)}")
    
    def _calculate_node_importance(self):
        """计算节点重要性"""
        logger.info("📊 计算节点重要性...")
        
        # 计算PageRank作为重要性指标
        try:
            pagerank = nx.pagerank(self.graph)
            self.node_importance = pagerank
        except:
            # 如果PageRank失败，使用度中心性
            degree_centrality = nx.degree_centrality(self.graph)
            self.node_importance = degree_centrality
        
        logger.info("✅ 节点重要性计算完成")
    
    def build_subgraph(self, seed_nodes: List[str], query: str) -> Dict[str, Any]:
        """基于重要性构建子图"""
        if not self.is_initialized:
            raise RuntimeError("图构建器未初始化")
        
        logger.info(f"🌱 基于重要性构建子图，种子节点: {len(seed_nodes)}")
        
        # 验证种子节点
        valid_seeds = [node for node in seed_nodes if node in self.graph.nodes]
        if not valid_seeds:
            logger.warning("⚠️ 没有有效的种子节点")
            return self._create_empty_subgraph()
        
        # 选择重要节点
        selected_nodes = set(valid_seeds)
        
        # 获取候选节点（种子节点的邻居及其邻居）
        candidates = set()
        for seed in valid_seeds:
            candidates.update(self.graph.neighbors(seed))
            for neighbor in self.graph.neighbors(seed):
                candidates.update(self.graph.neighbors(neighbor))
        
        # 根据重要性排序候选节点
        candidate_scores = []
        for candidate in candidates:
            if candidate not in selected_nodes:
                importance = self.node_importance.get(candidate, 0)
                candidate_scores.append((candidate, importance))
        
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的节点
        for candidate, score in candidate_scores:
            if len(selected_nodes) >= self.max_nodes:
                break
            if score >= self.importance_threshold:
                selected_nodes.add(candidate)
        
        # 提取子图
        subgraph = self._extract_subgraph(selected_nodes)
        
        logger.info(f"✅ 加权子图构建完成，节点: {len(subgraph['nodes'])}, 边: {len(subgraph['edges'])}")
        return subgraph
    
    def _extract_subgraph(self, nodes: Set[str]) -> Dict[str, Any]:
        """从完整图中提取子图"""
        subgraph = self.graph.subgraph(nodes)
        
        nodes_data = []
        edges_data = []
        
        for node_id in subgraph.nodes:
            node_data = subgraph.nodes[node_id].copy()
            node_data['id'] = node_id
            node_data['importance'] = self.node_importance.get(node_id, 0)
            nodes_data.append(node_data)
        
        for edge in subgraph.edges:
            edge_data = subgraph.edges[edge].copy()
            edge_data['source'] = edge[0]
            edge_data['target'] = edge[1]
            edges_data.append(edge_data)
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'node_count': len(nodes_data),
            'edge_count': len(edges_data),
            'algorithm': 'weighted'
        }
    
    def _create_empty_subgraph(self) -> Dict[str, Any]:
        """创建空子图"""
        return {
            'nodes': [],
            'edges': [],
            'node_count': 0,
            'edge_count': 0,
            'algorithm': 'weighted'
        }

# =============================================================================
# 注册所有图构建器
# =============================================================================

def register_all_graph_builders():
    """注册所有图构建器到工厂"""
    component_factory.register_graph_builder('pcst', PCSTGraphBuilder)
    component_factory.register_graph_builder('simple', SimpleGraphBuilder)
    component_factory.register_graph_builder('weighted', WeightedGraphBuilder)
    logger.info("✅ 所有图构建器已注册到组件工厂")

# 自动注册
register_all_graph_builders()
