"""
GNN处理器 (GNN Processor)
负责编排整个GNN处理流程：
1. 调用检索器获取种子节点
2. 调用GNN数据构建器准备torch_geometric数据
3. 调用GNN模型进行推理
4. 处理结果并返回
"""
import torch
import logging
from typing import Dict, Any, List, Optional
from app.rag.components import component_factory, ProcessorConfig, ComponentConfig, ProcessorDefaultConfigs
from app.rag.processors.base_processor import BaseProcessor

try:
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

logger = logging.getLogger(__name__)

class SimpleGNN(torch.nn.Module):
    """简单的GNN模型用于演示"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim) if HAS_TORCH_GEOMETRIC else None
        self.conv2 = GCNConv(hidden_dim, output_dim) if HAS_TORCH_GEOMETRIC else None
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, data):
        if not HAS_TORCH_GEOMETRIC:
            # 简化版本：直接返回节点特征的均值
            return torch.mean(data.x, dim=1) if data.x.size(0) > 0 else torch.zeros(128)
        
        x, edge_index = data.x, data.edge_index
        
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        # 全局池化（图级别表示）
        batch = torch.zeros(x.size(0), dtype=torch.long)  # 单图情况
        graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding

class GNNProcessor(BaseProcessor):
    """GNN处理器 - 编排GNN处理流程"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """初始化GNN处理器"""
        if config is None:
            config = ProcessorDefaultConfigs.get_gnn_processor_config()
        
        super().__init__(config)
        
        # GNN特有配置 - 从默认配置获取
        self.model_config = self._get_default_model_config()
        
        # GNN模型
        self.gnn_model = None
        
        if not HAS_TORCH_GEOMETRIC:
            logger.warning("⚠️ torch_geometric未安装，使用简化版GNN模型")
    
    def initialize(self) -> bool:
        """初始化GNN处理器"""
        try:
            # 调用父类初始化
            if not super().initialize():
                return False
            
            logger.info("🔄 初始化GNN特有组件...")
            
            # 创建GNN模型
            self.gnn_model = SimpleGNN(
                input_dim=self.model_config['input_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                output_dim=self.model_config['output_dim']
            )
            self.gnn_model.eval()  # 设置为评估模式
            
            logger.info("✅ GNN处理器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ GNN处理器初始化失败: {str(e)}")
            return False
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """具体处理逻辑"""
        logger.info(f"🧠 GNN处理器开始处理查询: {query[:50]}...")
        
        try:
            # 1. 检索种子节点
            logger.info("🔍 步骤1: 检索种子节点")
            retrieval_result = self.retriever.retrieve(query, top_k=5)
            seed_nodes = [item['node_id'] for item in retrieval_result]
            
            if not seed_nodes:
                logger.warning("⚠️ 未找到相关种子节点")
                return self._create_empty_result(query)
            
            logger.info(f"✅ 找到 {len(seed_nodes)} 个种子节点")
            
            # 2. 构建GNN数据 - 使用GNN数据构建器
            logger.info("🏗️ 步骤2: 构建GNN数据")
            gnn_data_result = self.graph_builder.build_subgraph(seed_nodes, query)
            
            if gnn_data_result['num_nodes'] == 0:
                logger.warning("⚠️ 构建的子图为空")
                return self._create_empty_result(query)
            
            logger.info(f"✅ 构建GNN数据完成，节点: {gnn_data_result['num_nodes']}, 边: {gnn_data_result['num_edges']}")
            
            # 3. GNN模型推理
            logger.info("🧠 步骤3: GNN模型推理")
            graph_embedding = self._run_gnn_inference(gnn_data_result['data'])
            
            # 4. 处理结果
            result = self._process_gnn_output(
                graph_embedding, 
                gnn_data_result, 
                retrieval_result, 
                query
            )
            
            logger.info("✅ GNN处理完成")
            return result
            
        except Exception as e:
            logger.error(f"❌ GNN处理失败: {str(e)}")
            return self._create_error_result(query, str(e))
    
    def _run_gnn_inference(self, data) -> torch.Tensor:
        """运行GNN推理"""
        with torch.no_grad():
            embedding = self.gnn_model(data)
            return embedding
    
    def _process_gnn_output(self, graph_embedding: torch.Tensor, 
                          gnn_data_result: Dict[str, Any],
                          retrieval_result: List[Dict[str, Any]], 
                          query: str) -> Dict[str, Any]:
        """处理GNN输出"""
        
        # 转换embedding为列表（用于JSON序列化）
        embedding_list = graph_embedding.squeeze().tolist()
        
        # 构建节点信息
        nodes_info = []
        for i, node_id in enumerate(gnn_data_result['node_ids']):
            node_info = {
                'node_id': node_id,
                'index': i,
                'type': node_id.split(':')[0] if ':' in node_id else 'unknown'
            }
            
            # 添加检索相关信息
            for item in retrieval_result:
                if item['node_id'] == node_id:
                    node_info.update({
                        'similarity_score': item.get('similarity_score', 0.0),
                        'description': item.get('description', ''),
                        'is_seed': True
                    })
                    break
            else:
                node_info['is_seed'] = False
            
            nodes_info.append(node_info)
        
        return {
            'success': True,
            'query': query,
            'processing_type': 'gnn',
            'graph_embedding': embedding_list,
            'embedding_dim': len(embedding_list),
            'subgraph_info': {
                'num_nodes': gnn_data_result['num_nodes'],
                'num_edges': gnn_data_result['num_edges'],
                'feature_dim': gnn_data_result['feature_dim'],
                'nodes': nodes_info
            },
            'seed_nodes': [item['node_id'] for item in retrieval_result],
            'metadata': {
                'retriever_type': self.config.retriever_config.component_name,
                'gnn_builder_type': self.config.graph_builder_config.component_name,
                'model_type': 'SimpleGNN'
            }
        }
    
    def _create_empty_result(self, query: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'success': True,
            'query': query,
            'processing_type': 'gnn',
            'graph_embedding': [],
            'embedding_dim': 0,
            'subgraph_info': {
                'num_nodes': 0,
                'num_edges': 0,
                'feature_dim': 0,
                'nodes': []
            },
            'seed_nodes': [],
            'metadata': {
                'message': 'No relevant data found'
            }
        }
    
    def _create_error_result(self, query: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'success': False,
            'query': query,
            'processing_type': 'gnn',
            'error': error_msg,
            'graph_embedding': [],
            'embedding_dim': 0,
            'subgraph_info': {
                'num_nodes': 0,
                'num_edges': 0,
                'feature_dim': 0,
                'nodes': []
            },
            'seed_nodes': [],
            'metadata': {}
        }
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """获取默认模型配置"""
        return {
            "input_dim": 768,
            "hidden_dim": 256,
            "output_dim": 128
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        stats = super().get_stats()
        stats.update({
            'model_params': {
                'input_dim': self.model_config.get('input_dim', 0),
                'hidden_dim': self.model_config.get('hidden_dim', 0),
                'output_dim': self.model_config.get('output_dim', 0)
            },
            'torch_geometric_available': HAS_TORCH_GEOMETRIC
        })
        return stats

# =============================================================================
# 工厂函数
# =============================================================================

def create_gnn_processor(custom_config: Optional[Dict[str, Any]] = None) -> GNNProcessor:
    """创建GNN处理器实例"""
    if custom_config:
        config = ProcessorDefaultConfigs.get_gnn_processor_config()
        
        # 更新配置
        if 'retriever' in custom_config:
            config.retriever_config.config.update(custom_config['retriever'])
        if 'graph_builder' in custom_config:
            config.graph_builder_config.config.update(custom_config['graph_builder'])
        if 'textualizer' in custom_config:
            config.textualizer_config.config.update(custom_config['textualizer'])
        
        for key in ['cache_enabled', 'cache_ttl', 'max_tokens']:
            if key in custom_config:
                setattr(config, key, custom_config[key])
        
        return GNNProcessor(config)
    else:
        return GNNProcessor()

# =============================================================================
# 注册GNN处理器
# =============================================================================

def register_gnn_processor():
    """注册GNN处理器到工厂"""
    try:
        component_factory.register_processor('gnn', GNNProcessor)
        logger.info("✅ GNN处理器已注册到组件工厂")
        
    except Exception as e:
        logger.error(f"❌ GNN处理器注册失败: {str(e)}")

# 自动注册
register_gnn_processor()
