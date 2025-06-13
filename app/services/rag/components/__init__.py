"""
RAG 组件模块初始化
自动注册所有组件到工厂
"""
import logging

logger = logging.getLogger(__name__)

# 导入所有组件模块以触发自动注册
from . import retrievers
from . import graph_builders
from . import textualizers
from . import gnn_data_builder
from . import graph_encoder

# 导出主要组件类
from .retrievers import (
    SemanticRetriever, 
    VectorRetriever, 
    KeywordRetriever, 
    HybridRetriever
)

from .graph_builders import (
    PCSTGraphBuilder, 
    SimpleGraphBuilder, 
    WeightedGraphBuilder
)

from .textualizers import (
    TemplateTextualizer, 
    CompactTextualizer, 
    QATextualizer
)

from .graph_encoder import (
    GraphEncoder,
    QAGraphEncoder,
    MultimodalContext,
    create_graph_encoder
)

__all__ = [
    # 检索器
    'SemanticRetriever',
    'VectorRetriever', 
    'KeywordRetriever',
    'HybridRetriever',
    
    # 图构建器
    'PCSTGraphBuilder',
    'SimpleGraphBuilder', 
    'WeightedGraphBuilder',
    
    # 文本化器
    'TemplateTextualizer',
    'CompactTextualizer',
    'QATextualizer',
    
    # 图编码器
    'GraphEncoder',
    'QAGraphEncoder',
    'MultimodalContext',
    'create_graph_encoder'
]

def initialize_components():
    """初始化所有RAG组件"""
    try:
        logger.info("🔧 正在初始化RAG组件模块...")
        
        # 导入并注册所有组件
        from . import retrievers
        from . import graph_builders  
        from . import textualizers
        from . import gnn_data_builder
        
        logger.info("✅ RAG组件模块初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG组件模块初始化失败: {str(e)}")
        return False

# 自动初始化
initialize_components()

# 导出主要类和工厂实例
from app.rag.component_factory import (
    BaseRetriever,
    BaseGraphBuilder, 
    BaseTextualizer,
    ComponentConfig,
    ProcessorConfig,
    ComponentFactory,
    component_factory,
    DefaultConfigs,
    ProcessorDefaultConfigs
)

# 导出具体组件类
from .retrievers import (
    SemanticRetriever,
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever
)

from .graph_builders import (
    PCSTGraphBuilder,
    SimpleGraphBuilder,
    WeightedGraphBuilder
)

from .gnn_data_builder import (
    GNNDataBuilder
)

from .textualizers import (
    TemplateTextualizer,
    CompactTextualizer,
    QATextualizer
)

__all__ = [
    # 基础接口
    'BaseRetriever',
    'BaseGraphBuilder',
    'BaseTextualizer',
    
    # 配置类
    'ComponentConfig',
    'ProcessorConfig',
    'DefaultConfigs',
    'ProcessorDefaultConfigs',
    
    # 工厂
    'ComponentFactory',
    'component_factory',
    
    # 检索器
    'SemanticRetriever',
    'VectorRetriever', 
    'KeywordRetriever',
    'HybridRetriever',
    
    # 图构建器
    'PCSTGraphBuilder',
    'SimpleGraphBuilder',
    'WeightedGraphBuilder',
    'GNNDataBuilder',
    
    # 文本化器
    'TemplateTextualizer',
    'CompactTextualizer',
    'QATextualizer'
]
