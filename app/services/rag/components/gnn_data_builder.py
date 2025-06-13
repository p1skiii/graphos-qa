"""
GNN数据构建器 (GNN Data Builder)
负责从NebulaGraph中提取子图数据并转换为torch_geometric.data.Data格式
适用于GNN模型的图神经网络处理
"""
import torch
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, Set
from app.database import NebulaGraphConnection
from app.rag.component_factory import BaseGraphBuilder, component_factory
import logging

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # 创建简单的替代Data类用于开发
    class Data:
        def __init__(self, x=None, edge_index=None, edge_attr=None, node_id=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.node_id = node_id

logger = logging.getLogger(__name__)

class GNNDataBuilder(BaseGraphBuilder):
    """GNN数据构建器 - 将子图转换为torch_geometric.data.Data格式"""
    
    def __init__(self, max_nodes: int = 50, max_hops: int = 2, 
                 feature_dim: int = 768, include_edge_features: bool = True, nebula_conn=None):
        """初始化GNN数据构建器
        
        Args:
            max_nodes: 最大节点数
            max_hops: 最大跳数
            feature_dim: 节点特征维度
            include_edge_features: 是否包含边特征
            nebula_conn: NebulaGraph连接实例
        """
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.feature_dim = feature_dim
        self.include_edge_features = include_edge_features
        
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.graph = nx.Graph()
        self.node_features = {}  # 节点特征缓存
        self.edge_features = {}  # 边特征缓存
        self.is_initialized = False
        
        # 检查torch_geometric可用性
        if not HAS_TORCH_GEOMETRIC:
            logger.warning("⚠️ torch_geometric未安装，使用简化版Data类")
    
    def initialize(self) -> bool:
        """初始化GNN数据构建器"""
        try:
            logger.info("🔄 初始化GNN数据构建器...")
            
            # 连接数据库
            if not self.nebula_conn.connect():
                logger.error("❌ NebulaGraph连接失败")
                return False
            
            # 构建基础图结构
            self._build_base_graph()
            
            self.is_initialized = True
            logger.info("✅ GNN数据构建器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ GNN数据构建器初始化失败: {str(e)}")
            return False
    
    def _build_base_graph(self):
        """构建基础图结构"""
        logger.info("🏗️ 构建基础图结构...")
        
        # 获取球员-球队关系数据
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
        
        logger.info(f"✅ 基础图构建完成，节点: {len(self.graph.nodes)}, 边: {len(self.graph.edges)}")
    
    def build_subgraph(self, seed_nodes: List[str], query: str) -> Dict[str, Any]:
        """构建子图并转换为GNN数据格式"""
        if not self.is_initialized:
            raise RuntimeError("GNN数据构建器未初始化")
        
        logger.info(f"🌱 构建GNN数据格式子图，种子节点: {len(seed_nodes)}")
        
        # 验证种子节点
        valid_seeds = [node for node in seed_nodes if node in self.graph.nodes]
        if not valid_seeds:
            logger.warning("⚠️ 没有有效的种子节点")
            return self._create_empty_gnn_data()
        
        try:
            # 1. 提取子图节点
            subgraph_nodes = self._extract_subgraph_nodes(valid_seeds)
            
            # 2. 构建节点特征
            node_features, node_mapping = self._build_node_features(subgraph_nodes, query)
            
            # 3. 构建边数据
            edge_index, edge_attr = self._build_edge_data(subgraph_nodes, node_mapping)
            
            # 4. 创建torch_geometric.data.Data对象
            data_obj = self._create_data_object(node_features, edge_index, edge_attr, 
                                              list(subgraph_nodes), node_mapping)
            
            # 5. 返回标准格式
            result = {
                'data': data_obj,
                'node_mapping': node_mapping,
                'node_ids': list(subgraph_nodes),
                'num_nodes': len(subgraph_nodes),
                'num_edges': edge_index.shape[1] if edge_index is not None else 0,
                'feature_dim': self.feature_dim,
                'algorithm': 'gnn_data_builder'
            }
            
            logger.info(f"✅ GNN数据构建完成，节点: {result['num_nodes']}, 边: {result['num_edges']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ GNN数据构建失败: {str(e)}")
            return self._create_empty_gnn_data()
    
    def _extract_subgraph_nodes(self, seed_nodes: List[str]) -> Set[str]:
        """提取子图节点"""
        # 验证种子节点
        valid_seeds = [node for node in seed_nodes if node in self.graph.nodes]
        if not valid_seeds:
            return set()
        
        # BFS扩展
        subgraph_nodes = set(valid_seeds)
        current_level = set(valid_seeds)
        
        for hop in range(self.max_hops):
            if len(subgraph_nodes) >= self.max_nodes:
                break
            
            next_level = set()
            for node in current_level:
                neighbors = list(self.graph.neighbors(node))
                
                # 限制每层添加的节点数
                remaining_slots = self.max_nodes - len(subgraph_nodes)
                neighbors = neighbors[:remaining_slots]
                
                for neighbor in neighbors:
                    if neighbor not in subgraph_nodes:
                        next_level.add(neighbor)
                        subgraph_nodes.add(neighbor)
                        
                        if len(subgraph_nodes) >= self.max_nodes:
                            break
                
                if len(subgraph_nodes) >= self.max_nodes:
                    break
            
            current_level = next_level
            if not current_level:
                break
        
        return subgraph_nodes
    
    def _build_node_features(self, nodes: Set[str], query: str) -> Tuple[torch.Tensor, Dict[str, int]]:
        """构建节点特征矩阵"""
        node_list = list(nodes)
        node_mapping = {node: i for i, node in enumerate(node_list)}
        
        # 构建特征矩阵
        features = []
        
        for node_id in node_list:
            node_data = self.graph.nodes[node_id]
            
            # 基础特征向量
            feature_vector = np.zeros(self.feature_dim)
            
            # 节点类型编码
            if node_data['type'] == 'player':
                feature_vector[0] = 1.0  # 球员标记
                feature_vector[1] = float(node_data.get('age', 0)) / 100.0  # 年龄归一化
            else:  # team
                feature_vector[2] = 1.0  # 球队标记
            
            # 度数特征
            degree = self.graph.degree(node_id)
            feature_vector[3] = float(degree) / 10.0  # 度数归一化
            
            # 查询相关特征（简化版本）
            query_features = self._compute_query_features(query, [node_id])
            if len(query_features) > 0:
                # 将查询特征加入特征向量的后面部分
                end_idx = min(len(query_features), self.feature_dim - 4)
                feature_vector[4:4+end_idx] = query_features[:end_idx]
            
            features.append(feature_vector)
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(np.array(features))
        
        return features_tensor, node_mapping
    
    def _compute_query_features(self, query: str, node_list: List[str]) -> np.ndarray:
        """计算查询相关特征"""
        # 简化实现：基于查询词匹配计算相似度
        query_lower = query.lower()
        features = []
        
        for node_id in node_list:
            node_data = self.graph.nodes[node_id]
            
            # 计算文本相似度
            text_content = f"{node_data.get('name', '')} {node_data.get('description', '')}"
            text_lower = text_content.lower()
            
            # 简单的关键词匹配分数
            query_words = query_lower.split()
            match_score = 0.0
            for word in query_words:
                if word in text_lower:
                    match_score += 1.0
            
            if len(query_words) > 0:
                match_score /= len(query_words)
            
            features.append(match_score)
        
        return np.array(features)
    
    def _build_edge_data(self, nodes: Set[str], node_mapping: Dict[str, int]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """构建边索引和边特征"""
        edge_list = []
        edge_attributes = []
        
        subgraph = self.graph.subgraph(nodes)
        
        for edge in subgraph.edges(data=True):
            source, target, edge_data = edge
            
            if source in node_mapping and target in node_mapping:
                source_idx = node_mapping[source]
                target_idx = node_mapping[target]
                
                # 添加双向边（无向图）
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])
                
                if self.include_edge_features:
                    # 边特征：权重、关系类型等
                    edge_feat = [
                        edge_data.get('weight', 1.0),
                        1.0 if edge_data.get('relation') == 'serve' else 0.0
                    ]
                    edge_attributes.append(edge_feat)
                    edge_attributes.append(edge_feat)  # 双向边使用相同特征
        
        if not edge_list:
            # 没有边的情况
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
        else:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_attr = torch.FloatTensor(edge_attributes) if self.include_edge_features else None
        
        return edge_index, edge_attr
    
    def _create_data_object(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                          edge_attr: Optional[torch.Tensor], node_ids: List[str], 
                          node_mapping: Dict[str, int]) -> Data:
        """创建torch_geometric.data.Data对象"""
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        # 添加额外信息
        data.node_id = node_ids
        data.num_nodes = len(node_ids)
        
        return data
    
    def _create_empty_gnn_data(self) -> Dict[str, Any]:
        """创建空的GNN数据"""
        empty_data = Data(
            x=torch.empty((0, self.feature_dim)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=None
        )
        empty_data.node_id = []
        empty_data.num_nodes = 0
        
        return {
            'data': empty_data,
            'node_mapping': {},
            'node_ids': [],
            'num_nodes': 0,
            'num_edges': 0,
            'feature_dim': self.feature_dim,
            'algorithm': 'gnn_data_builder'
        }

# =============================================================================
# 注册GNN数据构建器
# =============================================================================

def register_gnn_data_builder():
    """注册GNN数据构建器到工厂"""
    component_factory.register_graph_builder('gnn', GNNDataBuilder)
    logger.info("✅ GNN数据构建器已注册到组件工厂")

# 自动注册
register_gnn_data_builder()
