"""
G-Retriever 语义检索器
基于 G-Retriever 论文实现的语义检索模块
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.database import NebulaGraphConnection
from config import Config
import logging

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """G-Retriever 语义检索器"""
    
    def __init__(self):
        """初始化语义检索器"""
        self.embedding_model = None
        self.node_embeddings = {}  # 节点嵌入缓存
        self.edge_embeddings = {}  # 边嵌入缓存
        self.text_embeddings = {}  # 文本嵌入缓存
        self.nebula_conn = NebulaGraphConnection()
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            logger.info("🔄 正在初始化语义检索器...")
            
            # 加载嵌入模型
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info(f"✅ 嵌入模型加载成功: {Config.EMBEDDING_MODEL}")
            
            # 连接图数据库
            if not self.nebula_conn.connect():
                logger.error("❌ NebulaGraph连接失败")
                return False
                
            # 预计算节点和边的嵌入
            self._precompute_embeddings()
            
            self.is_initialized = True
            logger.info("✅ 语义检索器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 语义检索器初始化失败: {str(e)}")
            return False
    
    def _precompute_embeddings(self):
        """预计算节点和边的嵌入向量"""
        logger.info("🔢 正在预计算图元素嵌入向量...")
        
        # 计算球员节点嵌入
        self._compute_player_embeddings()
        
        # 计算球队节点嵌入
        self._compute_team_embeddings()
        
        # 计算边嵌入
        self._compute_edge_embeddings()
        
        logger.info(f"✅ 嵌入向量预计算完成，节点: {len(self.node_embeddings)}, 边: {len(self.edge_embeddings)}")
    
    def _compute_player_embeddings(self):
        """计算球员节点嵌入"""
        query = """
        MATCH (p:player)
        RETURN p.player.name AS name, p.player.age AS age
        LIMIT 100
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                name, age = row[0], row[1]
                
                # 构建球员描述文本
                description = f"球员 {name}, 年龄 {age} 岁"
                
                # 生成嵌入
                embedding = self.embedding_model.encode([description])[0]
                
                # 存储节点嵌入
                node_id = f"player:{name}"
                self.node_embeddings[node_id] = {
                    'embedding': embedding,
                    'type': 'player',
                    'name': name,
                    'description': description,
                    'properties': {'age': age}
                }
    
    def _compute_team_embeddings(self):
        """计算球队节点嵌入"""
        query = """
        MATCH (t:team)
        RETURN t.team.name AS name
        LIMIT 50
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                team_name = row[0]
                
                # 构建球队描述文本
                description = f"篮球队 {team_name}"
                
                # 生成嵌入
                embedding = self.embedding_model.encode([description])[0]
                
                # 存储节点嵌入
                node_id = f"team:{team_name}"
                self.node_embeddings[node_id] = {
                    'embedding': embedding,
                    'type': 'team',
                    'name': team_name,
                    'description': description,
                    'properties': {}
                }
    
    def _compute_edge_embeddings(self):
        """计算边嵌入"""
        query = """
        MATCH (p:player)-[s:serve]->(t:team)
        RETURN p.player.name AS player, t.team.name AS team
        LIMIT 100
        """
        
        result = self.nebula_conn.execute_query(query)
        if result['success']:
            for row in result['rows']:
                player_name, team_name = row[0], row[1]
                
                # 构建关系描述文本
                description = f"{player_name} 效力于 {team_name}"
                
                # 生成嵌入
                embedding = self.embedding_model.encode([description])[0]
                
                # 存储边嵌入
                edge_id = f"serve:{player_name}->{team_name}"
                self.edge_embeddings[edge_id] = {
                    'embedding': embedding,
                    'type': 'serve',
                    'source': f"player:{player_name}",
                    'target': f"team:{team_name}",
                    'description': description
                }
    
    def semantic_search_nodes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """基于语义相似度搜索相关节点"""
        if not self.is_initialized:
            logger.warning("检索器未初始化")
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算与所有节点的相似度
        similarities = []
        for node_id, node_data in self.node_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [node_data['embedding']]
            )[0][0]
            
            similarities.append({
                'node_id': node_id,
                'similarity': similarity,
                'type': node_data['type'],
                'name': node_data['name'],
                'description': node_data['description'],
                'properties': node_data['properties']
            })
        
        # 按相似度排序并返回top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def semantic_search_edges(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """基于语义相似度搜索相关边"""
        if not self.is_initialized:
            logger.warning("检索器未初始化")
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算与所有边的相似度
        similarities = []
        for edge_id, edge_data in self.edge_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [edge_data['embedding']]
            )[0][0]
            
            similarities.append({
                'edge_id': edge_id,
                'similarity': similarity,
                'type': edge_data['type'],
                'source': edge_data['source'],
                'target': edge_data['target'],
                'description': edge_data['description']
            })
        
        # 按相似度排序并返回top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_seed_nodes(self, query: str, top_k: int = 5) -> List[str]:
        """获取种子节点用于子图构建"""
        relevant_nodes = self.semantic_search_nodes(query, top_k)
        return [node['node_id'] for node in relevant_nodes if node['similarity'] > 0.3]
    
    def get_node_neighborhood(self, node_id: str, hop: int = 1) -> Dict[str, Any]:
        """获取节点的邻域信息"""
        # 解析节点类型和名称
        if node_id.startswith('player:'):
            node_type = 'player'
            node_name = node_id.replace('player:', '')
        elif node_id.startswith('team:'):
            node_type = 'team'
            node_name = node_id.replace('team:', '')
        else:
            return {'nodes': [], 'edges': []}
        
        # 构建查询
        if node_type == 'player':
            query = f"""
            MATCH (p:player {{name: '{node_name}'}})-[r]->(n)
            RETURN p.player.name as source, type(r) as rel_type, 
                   tags(n)[0] as target_type, properties(n) as target_props
            UNION
            MATCH (n)-[r]->(p:player {{name: '{node_name}'}})
            RETURN properties(n) as source_props, type(r) as rel_type,
                   'player' as target_type, p.player.name as target
            """
        else:  # team
            query = f"""
            MATCH (t:team {{name: '{node_name}'}})-[r]->(n)
            RETURN t.team.name as source, type(r) as rel_type,
                   tags(n)[0] as target_type, properties(n) as target_props
            UNION
            MATCH (n)-[r]->(t:team {{name: '{node_name}'}})
            RETURN properties(n) as source_props, type(r) as rel_type,
                   'team' as target_type, t.team.name as target
            """
        
        result = self.nebula_conn.execute_query(query)
        
        nodes = set()
        edges = []
        
        if result['success']:
            for row in result['rows']:
                # 处理查询结果，构建邻域图
                if len(row) >= 4:
                    source = row[0]
                    rel_type = row[1]
                    target_type = row[2]
                    target = row[3]
                    
                    # 添加节点
                    nodes.add(f"{node_type}:{node_name}")
                    if isinstance(target, str):
                        nodes.add(f"{target_type}:{target}")
                    
                    # 添加边
                    edges.append({
                        'source': f"{node_type}:{node_name}",
                        'target': f"{target_type}:{target}" if isinstance(target, str) else str(target),
                        'type': rel_type
                    })
        
        return {
            'nodes': list(nodes),
            'edges': edges
        }
    
    def close(self):
        """关闭连接"""
        if self.nebula_conn:
            self.nebula_conn.close()
