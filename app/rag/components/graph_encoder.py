"""
图编码器 (Graph Encoder)
专为QA任务设计的图到向量编码组件
将图结构转换为密集向量表示，用于ComplexGProcessor的增强模式
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # 创建简单替代Data类
    class Data:
        def __init__(self, x=None, edge_index=None, edge_attr=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr

logger = logging.getLogger(__name__)

# =============================================================================
# GNN模型 - 专为QA任务优化
# =============================================================================

class QAGraphEncoder(torch.nn.Module):
    """面向QA任务的图编码器"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if HAS_TORCH_GEOMETRIC:
            # 使用GCN层
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.2)
        else:
            # 简化版本 - 线性变换
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, data):
        """前向传播"""
        if not HAS_TORCH_GEOMETRIC:
            # 简化版本：节点特征的加权平均
            if hasattr(data, 'x') and data.x.size(0) > 0:
                x = self.linear1(data.x)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                # 返回图级别表示
                return torch.mean(x, dim=0)
            else:
                return torch.zeros(self.output_dim)
        
        x, edge_index = data.x, data.edge_index
        
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        # 全局池化获得图级别表示
        batch = torch.zeros(x.size(0), dtype=torch.long)
        graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding.squeeze()

# =============================================================================
# 图编码器组件
# =============================================================================

class GraphEncoder:
    """图编码器组件 - QA任务增强"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, 
                 output_dim: int = 128, device: str = 'cpu'):
        """初始化图编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            device: 计算设备
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # 创建模型
        self.model = QAGraphEncoder(input_dim, hidden_dim, output_dim)
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
        
        self.is_initialized = False
        logger.info(f"🧠 创建图编码器，输出维度: {output_dim}")
    
    def initialize(self) -> bool:
        """初始化图编码器"""
        try:
            logger.info("🔄 初始化图编码器...")
            
            # 检查torch_geometric可用性
            if not HAS_TORCH_GEOMETRIC:
                logger.warning("⚠️ torch_geometric未安装，使用简化版图编码器")
            
            self.is_initialized = True
            logger.info("✅ 图编码器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 图编码器初始化失败: {str(e)}")
            return False
    
    def encode_graph(self, graph_data, query: str = "") -> Dict[str, Any]:
        """将图数据编码为向量表示
        
        Args:
            graph_data: torch_geometric.data.Data对象或字典格式的图数据
            query: 查询文本（用于上下文）
            
        Returns:
            包含图嵌入的字典
        """
        if not self.is_initialized:
            raise RuntimeError("图编码器未初始化")
        
        try:
            logger.info("🔄 开始图编码...")
            
            # 如果输入是字典，先转换为Data对象
            if isinstance(graph_data, dict):
                graph_data = self._convert_subgraph_to_data(graph_data)
            
            with torch.no_grad():
                # 将数据移到指定设备
                if hasattr(graph_data, 'to'):
                    graph_data = graph_data.to(self.device)
                
                # 获取图嵌入
                graph_embedding = self.model(graph_data)
                
                # 转换为列表格式
                if isinstance(graph_embedding, torch.Tensor):
                    embedding_list = graph_embedding.cpu().numpy().tolist()
                else:
                    embedding_list = graph_embedding
                
                # 确保是列表格式
                if not isinstance(embedding_list, list):
                    embedding_list = [float(embedding_list)]
                
                result = {
                    'success': True,
                    'embedding': embedding_list,  # 改为embedding而不是graph_embedding
                    'embedding_dim': len(embedding_list),
                    'embedding_shape': f"({len(embedding_list)},)",
                    'model_type': 'QAGraphEncoder',
                    'query_context': query,
                    'graph_info': {
                        'num_nodes': graph_data.x.size(0) if hasattr(graph_data, 'x') and graph_data.x is not None else 0,
                        'num_edges': graph_data.edge_index.size(1) if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None else 0,
                        'feature_dim': graph_data.x.size(1) if hasattr(graph_data, 'x') and graph_data.x is not None else 0
                    }
                }
                
                logger.info(f"✅ 图编码完成，嵌入维度: {result['embedding_dim']}")
                return result
                
        except Exception as e:
            logger.error(f"❌ 图编码失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'embedding': [],
                'embedding_dim': 0,
                'query_context': query
            }
    
    def encode_subgraph_dict(self, subgraph: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """从子图字典直接编码（便捷方法）
        
        Args:
            subgraph: 包含nodes/edges的子图字典
            query: 查询文本
            
        Returns:
            包含图嵌入的字典
        """
        try:
            # 转换为torch_geometric.data.Data格式
            graph_data = self._convert_subgraph_to_data(subgraph)
            
            # 进行编码
            return self.encode_graph(graph_data, query)
            
        except Exception as e:
            logger.error(f"❌ 从子图字典编码失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'graph_embedding': [],
                'embedding_dim': 0,
                'query_context': query
            }
    
    def _convert_subgraph_to_data(self, subgraph: Dict[str, Any]) -> Data:
        """将子图字典转换为torch_geometric.data.Data格式"""
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        if not nodes:
            # 空图情况
            return Data(
                x=torch.empty((0, self.input_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=None
            )
        
        # 构建节点映射
        node_mapping = {node['id']: i for i, node in enumerate(nodes)}
        
        # 构建节点特征
        node_features = []
        for node in nodes:
            # 创建简单特征向量
            feature_vector = torch.zeros(self.input_dim)
            
            # 节点类型编码
            node_type = node.get('type', 'unknown')
            if node_type == 'player':
                feature_vector[0] = 1.0
                # 年龄特征
                age = node.get('age', 0)
                feature_vector[1] = float(age) / 100.0 if age else 0.0
            elif node_type == 'team':
                feature_vector[2] = 1.0
            
            node_features.append(feature_vector)
        
        # 构建边索引
        edge_list = []
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            if source in node_mapping and target in node_mapping:
                source_idx = node_mapping[source]
                target_idx = node_mapping[target]
                
                # 添加双向边
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])
        
        # 转换为tensor
        x = torch.stack(node_features) if node_features else torch.empty((0, self.input_dim))
        edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=None)
    
    def get_config(self) -> Dict[str, Any]:
        """获取编码器配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'device': self.device,
            'torch_geometric_available': HAS_TORCH_GEOMETRIC,
            'is_initialized': self.is_initialized
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取编码器统计信息"""
        return {
            'model_params': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            },
            'device': self.device,
            'torch_geometric_available': HAS_TORCH_GEOMETRIC,
            'is_initialized': self.is_initialized
        }

# =============================================================================
# 工厂函数
# =============================================================================

def create_graph_encoder(config: Optional[Dict[str, Any]] = None) -> GraphEncoder:
    """创建图编码器实例
    
    Args:
        config: 配置字典，可以包含model_config等嵌套配置
        
    Returns:
        GraphEncoder实例
    """
    if config is None:
        config = {}
    
    # 支持嵌套配置结构
    model_config = config.get('model_config', config)
    
    return GraphEncoder(
        input_dim=model_config.get('input_dim', 768),
        hidden_dim=model_config.get('hidden_dim', 256),
        output_dim=model_config.get('output_dim', 128),
        device=config.get('device', 'cpu')
    )

# =============================================================================
# 数据结构定义 - MultimodalContext
# =============================================================================

class MultimodalContext:
    """多模态上下文数据结构 - 包含文本和图嵌入"""
    
    def __init__(self, text_context: str, graph_embedding: List[float], 
                 metadata: Optional[Dict[str, Any]] = None):
        """初始化多模态上下文
        
        Args:
            text_context: 文本上下文
            graph_embedding: 图嵌入向量
            metadata: 元数据信息
        """
        self.text_context = text_context
        self.graph_embedding = graph_embedding
        self.metadata = metadata or {}
        
        # 验证数据
        if not isinstance(text_context, str):
            raise ValueError("text_context必须是字符串")
        if not isinstance(graph_embedding, list):
            raise ValueError("graph_embedding必须是列表")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text_context': self.text_context,
            'graph_embedding': self.graph_embedding,
            'embedding_dim': len(self.graph_embedding),
            'metadata': self.metadata,
            'modalities': ['text', 'graph'],
            'format_version': '1.0'
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultimodalContext':
        """从字典创建实例"""
        return cls(
            text_context=data['text_context'],
            graph_embedding=data['graph_embedding'],
            metadata=data.get('metadata', {})
        )
    
    def get_combined_representation(self) -> Dict[str, Any]:
        """获取组合表示（为LLM准备）"""
        return {
            'context_text': self.text_context,
            'graph_features': {
                'embedding': self.graph_embedding,
                'dimension': len(self.graph_embedding),
                'summary': f"图结构嵌入（{len(self.graph_embedding)}维向量）"
            },
            'integration_info': {
                'modality_count': 2,
                'text_length': len(self.text_context),
                'graph_dim': len(self.graph_embedding)
            },
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"MultimodalContext(text_len={len(self.text_context)}, graph_dim={len(self.graph_embedding)})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"MultimodalContext(text_context='{self.text_context[:50]}...', graph_embedding_dim={len(self.graph_embedding)}, metadata={self.metadata})"
