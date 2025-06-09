"""
RAG (检索增强生成) 核心功能
"""
import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.database import NebulaGraphConnection
from config import Config

class SimpleRAG:
    """简单的RAG实现"""
    
    def __init__(self):
        """初始化RAG系统"""
        self.embedding_model = None
        self.knowledge_base = []
        self.embeddings = None
        self.nebula_conn = NebulaGraphConnection()
        
    def initialize(self):
        """初始化模型和数据"""
        print("🔄 正在初始化RAG系统...")
        
        # 初始化嵌入模型
        try:
            print(f"📥 加载嵌入模型: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print("✅ 嵌入模型加载成功")
        except Exception as e:
            print(f"❌ 嵌入模型加载失败: {str(e)}")
            return False
            
        # 连接NebulaGraph
        if not self.nebula_conn.connect():
            print("❌ NebulaGraph连接失败")
            return False
            
        # 构建知识库
        self._build_knowledge_base()
        print("✅ RAG系统初始化完成")
        return True
        
    def _build_knowledge_base(self):
        """从NebulaGraph构建知识库"""
        print("🏗️  正在构建知识库...")
        
        # 获取球员信息
        players_query = """
        MATCH (p:player)
        RETURN p.player.name AS name, p.player.age AS age
        LIMIT 50
        """
        
        players_result = self.nebula_conn.execute_query(players_query)
        if players_result['success']:
            for row in players_result['rows']:
                name, age = row[0], row[1]
                text = f"球员 {name} 的年龄是 {age} 岁"
                self.knowledge_base.append({
                    'text': text,
                    'type': 'player_info',
                    'metadata': {'name': name, 'age': age}
                })
        
        # 获取球队信息
        teams_query = """
        MATCH (t:team)
        RETURN t.team.name AS name
        LIMIT 20
        """
        
        teams_result = self.nebula_conn.execute_query(teams_query)
        if teams_result['success']:
            for row in teams_result['rows']:
                team_name = row[0]
                text = f"球队 {team_name}"
                self.knowledge_base.append({
                    'text': text,
                    'type': 'team_info',
                    'metadata': {'name': team_name}
                })
        
        # 获取关系信息
        relations_query = """
        MATCH (p:player)-[s:serve]->(t:team)
        RETURN p.player.name AS player, t.team.name AS team
        LIMIT 30
        """
        
        relations_result = self.nebula_conn.execute_query(relations_query)
        if relations_result['success']:
            for row in relations_result['rows']:
                player_name, team_name = row[0], row[1]
                text = f"球员 {player_name} 效力于球队 {team_name}"
                self.knowledge_base.append({
                    'text': text,
                    'type': 'player_team_relation',
                    'metadata': {'player': player_name, 'team': team_name}
                })
        
        print(f"📚 知识库构建完成，共收集 {len(self.knowledge_base)} 条信息")
        
        # 生成嵌入向量
        if self.knowledge_base:
            texts = [item['text'] for item in self.knowledge_base]
            print("🔢 正在生成嵌入向量...")
            self.embeddings = self.embedding_model.encode(texts)
            print("✅ 嵌入向量生成完成")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关信息"""
        if not self.embedding_model or not self.embeddings.any():
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 获取top_k最相似的结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.knowledge_base[idx]['text'],
                'type': self.knowledge_base[idx]['type'],
                'metadata': self.knowledge_base[idx]['metadata'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题"""
        print(f"🤔 问题: {question}")
        
        # 检索相关信息
        retrieved_docs = self.retrieve(question, top_k=3)
        
        if not retrieved_docs:
            return {
                'answer': '抱歉，我没有找到相关信息。',
                'sources': [],
                'confidence': 0.0
            }
        
        # 简单的回答生成（基于检索到的信息）
        relevant_info = []
        for doc in retrieved_docs:
            if doc['similarity'] > 0.3:  # 相似度阈值
                relevant_info.append(doc['text'])
        
        if relevant_info:
            answer = "根据我的知识库，我找到以下相关信息：\n" + "\n".join(relevant_info)
            confidence = max([doc['similarity'] for doc in retrieved_docs])
        else:
            answer = "抱歉，我没有找到足够相关的信息来回答您的问题。"
            confidence = 0.0
        
        return {
            'answer': answer,
            'sources': retrieved_docs,
            'confidence': float(confidence)
        }
    
    def close(self):
        """关闭连接"""
        if self.nebula_conn:
            self.nebula_conn.close()

# 全局RAG实例
rag_system = SimpleRAG()
