"""
G-Retriever 图索引器
用于高效的节点索引和向量存储管理
"""
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from app.database import NebulaGraphConnection
from config import Config
import logging

logger = logging.getLogger(__name__)

class GraphIndexer:
    """图索引管理器"""
    
    def __init__(self, index_dir: str = "data/indexes"):
        """初始化索引器"""
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 核心组件
        self.embedding_model = None
        self.nebula_conn = NebulaGraphConnection()
        
        # 索引存储
        self.node_index = None      # FAISS索引
        self.edge_index = None      # FAISS索引
        self.node_metadata = {}     # 节点元数据
        self.edge_metadata = {}     # 边元数据
        self.id_to_index = {}       # ID到索引的映射
        self.index_to_id = {}       # 索引到ID的映射
        
        # 配置
        self.embedding_dim = 384    # sentence-transformers默认维度
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """初始化索引器"""
        try:
            logger.info("🔄 正在初始化图索引器...")
            
            # 加载嵌入模型
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ 嵌入模型加载成功，维度: {self.embedding_dim}")
            
            # 连接图数据库
            if not self.nebula_conn.connect():
                logger.error("❌ NebulaGraph连接失败")
                return False
            
            # 尝试加载现有索引
            if self._load_existing_indexes():
                logger.info("✅ 成功加载现有索引")
            else:
                logger.info("🏗️  开始构建新索引...")
                self._build_indexes()
                self._save_indexes()
                logger.info("✅ 新索引构建完成")
            
            self.is_initialized = True
            logger.info("✅ 图索引器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 图索引器初始化失败: {str(e)}")
            return False
    
    def _load_existing_indexes(self) -> bool:
        """加载现有索引"""
        try:
            # 检查索引文件是否存在
            node_index_path = self.index_dir / "node_index.faiss"
            edge_index_path = self.index_dir / "edge_index.faiss"
            node_metadata_path = self.index_dir / "node_metadata.pkl"
            edge_metadata_path = self.index_dir / "edge_metadata.pkl"
            mappings_path = self.index_dir / "mappings.json"
            
            if not all(p.exists() for p in [node_index_path, edge_index_path, 
                                           node_metadata_path, edge_metadata_path, 
                                           mappings_path]):
                return False
            
            # 加载FAISS索引
            self.node_index = faiss.read_index(str(node_index_path))
            self.edge_index = faiss.read_index(str(edge_index_path))
            
            # 加载元数据
            with open(node_metadata_path, 'rb') as f:
                self.node_metadata = pickle.load(f)
            
            with open(edge_metadata_path, 'rb') as f:
                self.edge_metadata = pickle.load(f)
            
            # 加载映射
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.id_to_index = mappings['id_to_index']
                self.index_to_id = mappings['index_to_id']
            
            logger.info(f"📚 索引加载完成 - 节点: {self.node_index.ntotal}, 边: {self.edge_index.ntotal}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  索引加载失败: {str(e)}")
            return False
    
    def _save_indexes(self):
        """保存索引到文件"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.node_index, str(self.index_dir / "node_index.faiss"))
            faiss.write_index(self.edge_index, str(self.index_dir / "edge_index.faiss"))
            
            # 保存元数据
            with open(self.index_dir / "node_metadata.pkl", 'wb') as f:
                pickle.dump(self.node_metadata, f)
            
            with open(self.index_dir / "edge_metadata.pkl", 'wb') as f:
                pickle.dump(self.edge_metadata, f)
            
            # 保存映射
            mappings = {
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }
            with open(self.index_dir / "mappings.json", 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
            
            logger.info("💾 索引保存完成")
            
        except Exception as e:
            logger.error(f"❌ 索引保存失败: {str(e)}")
    
    def _build_indexes(self):
        """构建新索引"""
        logger.info("🏗️  开始构建图索引...")
        
        # 构建节点索引
        self._build_node_index()
        
        # 构建边索引
        self._build_edge_index()
        
        logger.info("✅ 图索引构建完成")
    
    def _build_node_index(self):
        """构建节点索引"""
        logger.info("📊 构建节点索引...")
        
        # 获取所有节点
        nodes_data = []
        
        # 获取球员节点
        players_query = """
        MATCH (p:player)
        RETURN p.player.name AS name, p.player.age AS age
        LIMIT 200
        """
        
        result = self.nebula_conn.execute_query(players_query)
        if result['success']:
            for row in result['rows']:
                name, age = row[0], row[1]
                node_id = f"player:{name}"
                description = f"球员 {name}, 年龄 {age} 岁"
                
                nodes_data.append({
                    'node_id': node_id,  # 改为node_id以保持一致性
                    'id': node_id,      # 保留id作为备用
                    'type': 'player',
                    'name': name,
                    'description': description,
                    'properties': {'age': age}
                })
        
        # 获取球队节点
        teams_query = """
        MATCH (t:team)
        RETURN t.team.name AS name
        LIMIT 100
        """
        
        result = self.nebula_conn.execute_query(teams_query)
        if result['success']:
            for row in result['rows']:
                team_name = row[0]
                node_id = f"team:{team_name}"
                description = f"篮球队 {team_name}"
                
                nodes_data.append({
                    'node_id': node_id,  # 改为node_id以保持一致性
                    'id': node_id,      # 保留id作为备用
                    'type': 'team',
                    'name': team_name,
                    'description': description,
                    'properties': {}
                })
        
        # 生成嵌入并构建索引
        if nodes_data:
            descriptions = [node['description'] for node in nodes_data]
            embeddings = self.embedding_model.encode(descriptions)
            
            # 创建FAISS索引
            self.node_index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
            
            # 添加到索引
            self.node_index.add(embeddings.astype('float32'))
            
            # 存储元数据和映射
            for i, node in enumerate(nodes_data):
                self.node_metadata[i] = node
                self.id_to_index[node['id']] = i
                self.index_to_id[str(i)] = node['id']
        
        logger.info(f"✅ 节点索引构建完成，共 {len(nodes_data)} 个节点")
    
    def _build_edge_index(self):
        """构建边索引"""
        logger.info("🔗 构建边索引...")
        
        # 获取所有边
        edges_data = []
        
        # 获取效力关系
        serve_query = """
        MATCH (p:player)-[s:serve]->(t:team)
        RETURN p.player.name AS player, t.team.name AS team
        LIMIT 300
        """
        
        result = self.nebula_conn.execute_query(serve_query)
        if result['success']:
            for row in result['rows']:
                player_name, team_name = row[0], row[1]
                edge_id = f"serve:{player_name}->{team_name}"
                description = f"{player_name} 效力于 {team_name}"
                
                edges_data.append({
                    'edge_id': edge_id,  # 改为edge_id以保持一致性
                    'id': edge_id,      # 保留id作为备用
                    'type': 'serve',
                    'source': f"player:{player_name}",
                    'target': f"team:{team_name}",
                    'description': description,
                    'properties': {}
                })
        
        # 生成嵌入并构建索引
        if edges_data:
            descriptions = [edge['description'] for edge in edges_data]
            embeddings = self.embedding_model.encode(descriptions)
            
            # 创建FAISS索引
            self.edge_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # 添加到索引
            self.edge_index.add(embeddings.astype('float32'))
            
            # 存储元数据
            for i, edge in enumerate(edges_data):
                edge_idx = f"edge_{i}"
                self.edge_metadata[edge_idx] = edge
        
        logger.info(f"✅ 边索引构建完成，共 {len(edges_data)} 条边")
    
    def search_nodes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相关节点"""
        if not self.is_initialized or self.node_index is None:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # 搜索
        scores, indices = self.node_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx in self.node_metadata:
                node_data = self.node_metadata[idx].copy()
                node_data['similarity'] = float(score)
                node_data['rank'] = i + 1
                results.append(node_data)
        
        return results
    
    def search_edges(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相关边"""
        if not self.is_initialized or self.edge_index is None:
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # 搜索
        scores, indices = self.edge_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            edge_key = f"edge_{idx}"
            if edge_key in self.edge_metadata:
                edge_data = self.edge_metadata[edge_key].copy()
                edge_data['similarity'] = float(score)
                edge_data['rank'] = i + 1
                results.append(edge_data)
        
        return results
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取节点信息"""
        if node_id in self.id_to_index:
            idx = self.id_to_index[node_id]
            if idx in self.node_metadata:
                return self.node_metadata[idx]
        return None
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """根据类型获取节点"""
        results = []
        for node_data in self.node_metadata.values():
            if node_data.get('type') == node_type:
                results.append(node_data)
        return results
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """获取节点的邻居"""
        neighbors = []
        
        # 在边数据中查找包含该节点的边
        for edge_data in self.edge_metadata.values():
            source = edge_data.get('source')
            target = edge_data.get('target')
            
            if source == node_id and target not in neighbors:
                neighbors.append(target)
            elif target == node_id and source not in neighbors:
                neighbors.append(source)
        
        return neighbors
    
    def rebuild_index(self):
        """重建索引"""
        logger.info("🔄 重建索引中...")
        
        # 清空现有索引
        self.node_index = None
        self.edge_index = None
        self.node_metadata.clear()
        self.edge_metadata.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        # 重新构建
        self._build_indexes()
        self._save_indexes()
        
        logger.info("✅ 索引重建完成")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        stats = {
            'initialized': self.is_initialized,
            'embedding_dim': self.embedding_dim,
            'node_count': self.node_index.ntotal if self.node_index else 0,
            'edge_count': self.edge_index.ntotal if self.edge_index else 0,
            'node_types': {},
            'edge_types': {}
        }
        
        # 统计节点类型
        for node_data in self.node_metadata.values():
            node_type = node_data.get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # 统计边类型
        for edge_data in self.edge_metadata.values():
            edge_type = edge_data.get('type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        return stats
    
    def close(self):
        """关闭连接"""
        if self.nebula_conn:
            self.nebula_conn.close()
