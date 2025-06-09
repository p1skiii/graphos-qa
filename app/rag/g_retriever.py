"""
G-Retriever 主系统
整合语义检索、图构建、上下文格式化的完整RAG系统
"""
from typing import List, Dict, Any, Optional
from .semantic_retriever import SemanticRetriever
from .graph_constructor import GraphConstructor
from .context_formatter import ContextFormatter
from .graph_indexer import GraphIndexer
import logging
import time

logger = logging.getLogger(__name__)

class GRetriever:
    """G-Retriever 主系统"""
    
    def __init__(self):
        """初始化G-Retriever系统"""
        # 核心组件
        self.semantic_retriever = SemanticRetriever()
        self.graph_constructor = GraphConstructor()
        self.context_formatter = ContextFormatter()
        self.graph_indexer = GraphIndexer()
        
        # 状态标志
        self.is_initialized = False
        
        # 配置参数
        self.config = {
            'max_seed_nodes': 5,
            'max_subgraph_nodes': 20,
            'similarity_threshold': 0.3,
            'context_format': 'qa'  # 'qa', 'detailed', 'compact'
        }
    
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("🚀 正在初始化G-Retriever系统...")
            start_time = time.time()
            
            # 1. 初始化图索引器（最重要的组件）
            logger.info("📊 初始化图索引器...")
            if not self.graph_indexer.initialize():
                logger.error("❌ 图索引器初始化失败")
                return False
            
            # 2. 初始化语义检索器
            logger.info("🔍 初始化语义检索器...")
            if not self.semantic_retriever.initialize():
                logger.error("❌ 语义检索器初始化失败")
                return False
            
            # 3. 初始化图构造器
            logger.info("🏗️  初始化图构造器...")
            if not self.graph_constructor.initialize():
                logger.error("❌ 图构造器初始化失败")
                return False
            
            self.is_initialized = True
            elapsed_time = time.time() - start_time
            logger.info(f"✅ G-Retriever系统初始化完成，耗时: {elapsed_time:.2f}秒")
            
            # 打印系统统计信息
            self._print_system_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ G-Retriever系统初始化失败: {str(e)}")
            return False
    
    def retrieve_and_answer(self, query: str, format_type: str = 'qa') -> Dict[str, Any]:
        """完整的检索和回答流程"""
        if not self.is_initialized:
            return {
                'answer': '系统未初始化，请先初始化系统。',
                'subgraph_info': {},
                'context': '',
                'confidence': 0.0,
                'processing_time': 0.0
            }
        
        start_time = time.time()
        logger.info(f"🤔 开始处理查询: {query}")
        
        try:
            # 步骤1: 语义检索获取种子节点
            logger.info("🌱 步骤1: 获取种子节点...")
            seed_nodes = self._get_seed_nodes(query)
            logger.info(f"找到 {len(seed_nodes)} 个种子节点: {seed_nodes}")
            
            if not seed_nodes:
                return {
                    'answer': '抱歉，没有找到相关的图结构信息。',
                    'subgraph_info': {},
                    'context': '',
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time
                }
            
            # 步骤2: 设置节点奖励和边成本
            logger.info("💰 步骤2: 设置节点奖励和边成本...")
            self._set_graph_weights(query)
            
            # 步骤3: 构建子图
            logger.info("🏗️  步骤3: 构建子图...")
            subgraph = self.graph_constructor.pcst_subgraph(
                seed_nodes, 
                max_nodes=self.config['max_subgraph_nodes']
            )
            
            if subgraph.number_of_nodes() == 0:
                return {
                    'answer': '无法构建有效的子图结构。',
                    'subgraph_info': {},
                    'context': '',
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time
                }
            
            # 步骤4: 提取子图信息
            logger.info("📊 步骤4: 提取子图信息...")
            subgraph_info = self.graph_constructor.extract_subgraph_info(subgraph)
            
            # 步骤5: 格式化上下文
            logger.info("📝 步骤5: 格式化上下文...")
            context = self._format_context(subgraph_info, query, format_type)
            
            # 步骤6: 生成答案
            logger.info("💡 步骤6: 生成答案...")
            answer, confidence = self._generate_answer(subgraph_info, query, context)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ 查询处理完成，耗时: {processing_time:.2f}秒，置信度: {confidence:.2f}")
            
            return {
                'answer': answer,
                'subgraph_info': subgraph_info,
                'context': context,
                'confidence': confidence,
                'processing_time': processing_time,
                'seed_nodes': seed_nodes
            }
            
        except Exception as e:
            logger.error(f"❌ 查询处理失败: {str(e)}")
            return {
                'answer': f'处理查询时发生错误: {str(e)}',
                'subgraph_info': {},
                'context': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _get_seed_nodes(self, query: str) -> List[str]:
        """获取种子节点"""
        # 使用图索引器进行快速检索
        relevant_nodes = self.graph_indexer.search_nodes(
            query, 
            top_k=self.config['max_seed_nodes'] * 2
        )
        
        # 过滤高相似度节点
        seed_nodes = []
        for node in relevant_nodes:
            if node['similarity'] > self.config['similarity_threshold']:
                seed_nodes.append(node['id'])
                if len(seed_nodes) >= self.config['max_seed_nodes']:
                    break
        
        return seed_nodes
    
    def _set_graph_weights(self, query: str):
        """设置图权重"""
        # 获取相关节点和边
        relevant_nodes = self.graph_indexer.search_nodes(query, top_k=20)
        relevant_edges = self.graph_indexer.search_edges(query, top_k=20)
        
        # 设置节点奖励
        self.graph_constructor.set_node_prizes(relevant_nodes)
        
        # 设置边成本
        self.graph_constructor.set_edge_costs(relevant_edges)
    
    def _format_context(self, subgraph_info: Dict[str, Any], query: str, format_type: str) -> str:
        """格式化上下文"""
        if format_type == 'detailed':
            return self.context_formatter.format_subgraph_to_text(subgraph_info, query)
        elif format_type == 'compact':
            return self.context_formatter.format_subgraph_compact(subgraph_info)
        else:  # 'qa'
            return self.context_formatter.format_subgraph_to_qa_context(subgraph_info, query)
    
    def _generate_answer(self, subgraph_info: Dict[str, Any], query: str, context: str) -> tuple:
        """生成答案"""
        nodes = subgraph_info.get('nodes', [])
        edges = subgraph_info.get('edges', [])
        
        if not nodes and not edges:
            return "抱歉，没有找到相关信息。", 0.0
        
        # 简单的规则基础答案生成
        query_lower = query.lower()
        answer_parts = []
        confidence = 0.0
        
        # 查找直接匹配的实体
        mentioned_entities = []
        for node in nodes:
            name = node.get('name', '').lower()
            if name and name in query_lower:
                mentioned_entities.append(node)
                confidence = max(confidence, 0.8)
        
        if mentioned_entities:
            # 生成基于实体的答案
            for entity in mentioned_entities:
                entity_type = entity.get('type', '')
                name = entity.get('name', '')
                props = entity.get('properties', {})
                
                if entity_type == 'player':
                    age = props.get('age')
                    if '年龄' in query_lower or 'age' in query_lower:
                        if age:
                            answer_parts.append(f"{name}的年龄是{age}岁。")
                            confidence = max(confidence, 0.9)
                    else:
                        answer_parts.append(f"{name}是一名篮球球员。")
                elif entity_type == 'team':
                    answer_parts.append(f"{name}是一支篮球队。")
        
        # 查找效力关系
        if '效力' in query_lower or '打球' in query_lower or '在哪' in query_lower:
            serve_edges = [e for e in edges if e.get('type') == 'serve']
            for edge in serve_edges:
                player_name = self.context_formatter._extract_name_from_id(edge.get('source', ''))
                team_name = self.context_formatter._extract_name_from_id(edge.get('target', ''))
                
                if player_name.lower() in query_lower:
                    answer_parts.append(f"{player_name}效力于{team_name}。")
                    confidence = max(confidence, 0.85)
        
        # 如果没有生成具体答案，使用上下文信息
        if not answer_parts:
            if context:
                answer_parts.append("根据图结构信息，我找到以下相关内容：")
                answer_parts.append(context)
                confidence = 0.6
            else:
                answer_parts.append("抱歉，无法根据现有信息回答您的问题。")
                confidence = 0.1
        
        final_answer = "\n".join(answer_parts)
        
        # 添加补充信息
        if len(nodes) > 0:
            final_answer += f"\n\n（基于 {len(nodes)} 个相关实体和 {len(edges)} 个关系构建的答案）"
        
        return final_answer, confidence
    
    def _print_system_stats(self):
        """打印系统统计信息"""
        try:
            stats = self.graph_indexer.get_index_stats()
            logger.info("📈 系统统计信息:")
            logger.info(f"  - 节点总数: {stats['node_count']}")
            logger.info(f"  - 边总数: {stats['edge_count']}")
            logger.info(f"  - 嵌入维度: {stats['embedding_dim']}")
            
            if stats['node_types']:
                logger.info("  - 节点类型分布:")
                for node_type, count in stats['node_types'].items():
                    logger.info(f"    * {node_type}: {count}")
            
            if stats['edge_types']:
                logger.info("  - 边类型分布:")
                for edge_type, count in stats['edge_types'].items():
                    logger.info(f"    * {edge_type}: {count}")
                    
        except Exception as e:
            logger.warning(f"⚠️  获取统计信息失败: {str(e)}")
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"📝 配置更新: {key} = {value}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'initialized': self.is_initialized,
            'config': self.config.copy()
        }
        
        if self.is_initialized:
            try:
                info['stats'] = self.graph_indexer.get_index_stats()
            except:
                info['stats'] = {}
        
        return info
    
    def close(self):
        """关闭所有连接"""
        logger.info("🔌 正在关闭G-Retriever系统...")
        
        if hasattr(self, 'semantic_retriever'):
            self.semantic_retriever.close()
        
        if hasattr(self, 'graph_constructor'):
            self.graph_constructor.close()
        
        if hasattr(self, 'graph_indexer'):
            self.graph_indexer.close()
        
        logger.info("✅ G-Retriever系统已关闭")

# 全局G-Retriever实例
g_retriever_system = GRetriever()
