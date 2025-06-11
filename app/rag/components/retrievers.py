"""
RAG 检索器组件集合
实现多种检索策略：语义检索、向量检索、关键词检索、混合检索
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging
from app.database import NebulaGraphConnection
from config import Config
from app.rag.component_factory import BaseRetriever, component_factory

logger = logging.getLogger(__name__)

# =============================================================================
# 语义检索器
# =============================================================================

class SemanticRetriever(BaseRetriever):
    """语义检索器 - 基于句子嵌入的相似度检索"""
    
    def __init__(self, embedding_model: str = None, top_k: int = 5, 
                 similarity_threshold: float = 0.3, nebula_conn=None):
        """初始化语义检索器"""
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.embedding_model = None
        self.node_embeddings = {}
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            logger.info(f"🔄 初始化语义检索器 ({self.embedding_model_name})...")
            
            # 加载嵌入模型
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 预计算节点嵌入
            self._precompute_embeddings()
            
            self.is_initialized = True
            logger.info("✅ 语义检索器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 语义检索器初始化失败: {str(e)}")
            return False
    
    def _precompute_embeddings(self):
        """预计算节点嵌入向量"""
        logger.info("🔢 预计算节点嵌入向量...")
        
        # 计算球员节点嵌入
        self._compute_player_embeddings()
        
        # 计算球队节点嵌入
        self._compute_team_embeddings()
        
        logger.info(f"✅ 嵌入向量计算完成，节点数: {len(self.node_embeddings)}")
    
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
                # 处理可能的NULL值
                if name is None:
                    continue
                if age is None:
                    age = 0
                    
                description = f"球员 {name}, 年龄 {age} 岁"
                
                embedding = self.embedding_model.encode([description])[0]
                
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
                name = row[0]
                # 处理可能的NULL值
                if name is None:
                    continue
                    
                description = f"球队 {name}"
                
                embedding = self.embedding_model.encode([description])[0]
                
                node_id = f"team:{name}"
                self.node_embeddings[node_id] = {
                    'embedding': embedding,
                    'type': 'team',
                    'name': name,
                    'description': description,
                    'properties': {}
                }
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """语义检索相关节点"""
        if not self.is_initialized:
            raise RuntimeError("检索器未初始化")
        
        top_k = top_k or self.top_k
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = []
        for node_id, node_data in self.node_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding], 
                [node_data['embedding']]
            )[0][0]
            
            if similarity >= self.similarity_threshold:
                similarities.append({
                    'node_id': node_id,
                    'similarity': float(similarity),
                    'type': node_data['type'],
                    'name': node_data['name'],
                    'description': node_data['description'],
                    'properties': node_data['properties']
                })
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

# =============================================================================
# 向量检索器
# =============================================================================

class VectorRetriever(BaseRetriever):
    """向量检索器 - 优化的向量相似度检索"""
    
    def __init__(self, embedding_model: str = None, top_k: int = 5, 
                 use_faiss: bool = False, nebula_conn=None):
        """初始化向量检索器"""
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.top_k = top_k
        self.use_faiss = use_faiss
        
        self.embedding_model = None
        self.vectors = None
        self.node_index = {}
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            logger.info(f"🔄 初始化向量检索器...")
            
            # 加载嵌入模型
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 构建向量索引
            self._build_vector_index()
            
            self.is_initialized = True
            logger.info("✅ 向量检索器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 向量检索器初始化失败: {str(e)}")
            return False
    
    def _build_vector_index(self):
        """构建向量索引"""
        logger.info("🔢 构建向量索引...")
        
        nodes_data = []
        embeddings = []
        
        # 获取所有节点
        queries = [
            "MATCH (p:player) RETURN 'player' AS type, p.player.name AS name, p.player.age AS age LIMIT 100",
            "MATCH (t:team) RETURN 'team' AS type, t.team.name AS name, NULL AS age LIMIT 50"
        ]
        
        for query in queries:
            result = self.nebula_conn.execute_query(query)
            if result['success']:
                for row in result['rows']:
                    node_type, name, age = row[0], row[1], row[2]
                    
                    # 处理NULL值
                    if name is None:
                        continue
                    if age is None:
                        age = 0
                    
                    # 构建描述
                    if node_type == 'player':
                        description = f"球员 {name}, 年龄 {age} 岁"
                    else:
                        description = f"球队 {name}"
                    
                    # 生成嵌入
                    embedding = self.embedding_model.encode([description])[0]
                    
                    node_data = {
                        'node_id': f"{node_type}:{name}",
                        'type': node_type,
                        'name': name,
                        'description': description,
                        'properties': {'age': age} if age else {}
                    }
                    
                    nodes_data.append(node_data)
                    embeddings.append(embedding)
        
        # 转换为numpy数组
        self.vectors = np.array(embeddings)
        
        # 构建索引映射
        for i, node_data in enumerate(nodes_data):
            self.node_index[i] = node_data
        
        logger.info(f"✅ 向量索引构建完成，节点数: {len(self.node_index)}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """向量检索相关节点"""
        if not self.is_initialized:
            raise RuntimeError("检索器未初始化")
        
        top_k = top_k or self.top_k
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.vectors)[0]
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            node_data = self.node_index[idx].copy()
            node_data['similarity'] = float(similarities[idx])
            results.append(node_data)
        
        return results

# =============================================================================
# 关键词检索器
# =============================================================================

class KeywordRetriever(BaseRetriever):
    """关键词检索器 - 基于TF-IDF的关键词匹配"""
    
    def __init__(self, top_k: int = 5, use_tfidf: bool = True, 
                 min_score: float = 0.1, nebula_conn=None):
        """初始化关键词检索器"""
        self.top_k = top_k
        self.use_tfidf = use_tfidf
        self.min_score = min_score
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.node_index = {}
        # 使用传递的连接或创建新连接
        self.nebula_conn = nebula_conn if nebula_conn is not None else NebulaGraphConnection()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            logger.info("🔄 初始化关键词检索器...")
            
            # 检查数据库连接
            if not self.nebula_conn.session:
                if not self.nebula_conn.connect():
                    logger.error("❌ NebulaGraph连接失败")
                    return False
            
            # 构建TF-IDF索引
            self._build_tfidf_index()
            
            self.is_initialized = True
            logger.info("✅ 关键词检索器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 关键词检索器初始化失败: {str(e)}")
            return False
    
    def _build_tfidf_index(self):
        """构建TF-IDF索引"""
        logger.info("📝 构建TF-IDF索引...")
        
        documents = []
        
        # 获取所有节点文档
        queries = [
            "MATCH (p:player) RETURN 'player' AS type, p.player.name AS name, p.player.age AS age LIMIT 100",
            "MATCH (t:team) RETURN 'team' AS type, t.team.name AS name, NULL AS age LIMIT 50"
        ]
        
        doc_idx = 0
        for query in queries:
            result = self.nebula_conn.execute_query(query)
            if result['success']:
                for row in result['rows']:
                    node_type, name, age = row[0], row[1], row[2]
                    
                    # 处理NULL值
                    if name is None:
                        continue
                    if age is None:
                        age = 0
                    
                    # 构建文档文本
                    if node_type == 'player':
                        document = f"球员 {name} 年龄 {age} 岁"
                        keywords = [name, str(age), '球员', '年龄']
                    else:
                        document = f"球队 {name}"
                        keywords = [name, '球队']
                    
                    documents.append(document)
                    
                    self.node_index[doc_idx] = {
                        'node_id': f"{node_type}:{name}",
                        'type': node_type,
                        'name': name,
                        'document': document,
                        'keywords': keywords,
                        'properties': {'age': age} if age else {}
                    }
                    doc_idx += 1
        
        self.documents = documents
        
        # 构建TF-IDF矩阵
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                lowercase=True
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        logger.info(f"✅ TF-IDF索引构建完成，文档数: {len(documents)}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """关键词检索相关节点"""
        if not self.is_initialized:
            raise RuntimeError("检索器未初始化")
        
        top_k = top_k or self.top_k
        
        if self.use_tfidf:
            return self._tfidf_retrieve(query, top_k)
        else:
            return self._keyword_match_retrieve(query, top_k)
    
    def _tfidf_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TF-IDF检索"""
        # 转换查询
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.min_score:
                node_data = self.node_index[idx].copy()
                node_data['similarity'] = float(similarities[idx])
                results.append(node_data)
        
        return results
    
    def _keyword_match_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """简单关键词匹配"""
        query_keywords = re.findall(r'\w+', query.lower())
        
        scores = []
        for idx, node_data in self.node_index.items():
            score = 0
            for keyword in query_keywords:
                if any(keyword in str(k).lower() for k in node_data['keywords']):
                    score += 1
            
            if score > 0:
                scores.append({
                    'index': idx,
                    'score': score / len(query_keywords)
                })
        
        # 排序并返回top_k
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for item in scores[:top_k]:
            if item['score'] >= self.min_score:
                node_data = self.node_index[item['index']].copy()
                node_data['similarity'] = float(item['score'])
                results.append(node_data)
        
        return results

# =============================================================================
# 混合检索器
# =============================================================================

class HybridRetriever(BaseRetriever):
    """混合检索器 - 结合多种检索策略"""
    
    def __init__(self, semantic_weight: float = 0.5, keyword_weight: float = 0.3, 
                 vector_weight: float = 0.2, top_k: int = 5, nebula_conn=None):
        """初始化混合检索器"""
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.top_k = top_k
        
        # 传递连接给子检索器
        self.semantic_retriever = SemanticRetriever(nebula_conn=nebula_conn)
        self.keyword_retriever = KeywordRetriever(nebula_conn=nebula_conn)
        self.vector_retriever = VectorRetriever(nebula_conn=nebula_conn)
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            logger.info("🔄 初始化混合检索器...")
            
            # 初始化所有子检索器
            if not self.semantic_retriever.initialize():
                logger.error("❌ 语义检索器初始化失败")
                return False
            
            if not self.keyword_retriever.initialize():
                logger.error("❌ 关键词检索器初始化失败")
                return False
            
            if not self.vector_retriever.initialize():
                logger.error("❌ 向量检索器初始化失败")
                return False
            
            self.is_initialized = True
            logger.info("✅ 混合检索器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 混合检索器初始化失败: {str(e)}")
            return False
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """混合检索相关节点"""
        if not self.is_initialized:
            raise RuntimeError("检索器未初始化")
        
        top_k = top_k or self.top_k
        
        # 从各个检索器获取结果
        semantic_results = self.semantic_retriever.retrieve(query, top_k * 2)
        keyword_results = self.keyword_retriever.retrieve(query, top_k * 2)
        vector_results = self.vector_retriever.retrieve(query, top_k * 2)
        
        # 合并和重排序
        combined_scores = {}
        
        # 处理语义检索结果
        for result in semantic_results:
            node_id = result['node_id']
            combined_scores[node_id] = combined_scores.get(node_id, 0) + \
                                     result['similarity'] * self.semantic_weight
        
        # 处理关键词检索结果
        for result in keyword_results:
            node_id = result['node_id']
            combined_scores[node_id] = combined_scores.get(node_id, 0) + \
                                     result['similarity'] * self.keyword_weight
        
        # 处理向量检索结果
        for result in vector_results:
            node_id = result['node_id']
            combined_scores[node_id] = combined_scores.get(node_id, 0) + \
                                     result['similarity'] * self.vector_weight
        
        # 创建节点数据映射
        all_results = {}
        for results in [semantic_results, keyword_results, vector_results]:
            for result in results:
                all_results[result['node_id']] = result
        
        # 排序并返回top_k
        sorted_nodes = sorted(combined_scores.items(), 
                            key=lambda x: x[1], reverse=True)
        
        final_results = []
        for node_id, score in sorted_nodes[:top_k]:
            if node_id in all_results:
                node_data = all_results[node_id].copy()
                node_data['similarity'] = float(score)
                final_results.append(node_data)
        
        return final_results

# =============================================================================
# 注册所有检索器
# =============================================================================

def register_all_retrievers():
    """注册所有检索器到工厂"""
    component_factory.register_retriever('semantic', SemanticRetriever)
    component_factory.register_retriever('vector', VectorRetriever)
    component_factory.register_retriever('keyword', KeywordRetriever)
    component_factory.register_retriever('hybrid', HybridRetriever)
    logger.info("✅ 所有检索器已注册到组件工厂")

# 自动注册
register_all_retrievers()
