"""
RAG 组件工厂系统
实现可插拔的组件架构，支持不同类型的检索器、图构建器和文本化器
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass
import logging

# 导入全局连接
from app.database.nebula_connection import nebula_conn

logger = logging.getLogger(__name__)

# =============================================================================
# 基础接口定义
# =============================================================================

class BaseRetriever(ABC):
    """检索器基础接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化检索器"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关节点"""
        pass

class BaseGraphBuilder(ABC):
    """图构建器基础接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化图构建器"""
        pass
    
    @abstractmethod
    def build_subgraph(self, seed_nodes: List[str], query: str) -> Dict[str, Any]:
        """构建子图"""
        pass

class BaseTextualizer(ABC):
    """文本化器基础接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化文本化器"""
        pass
    
    @abstractmethod
    def textualize(self, subgraph: Dict[str, Any], query: str) -> str:
        """将子图转换为文本"""
        pass

# =============================================================================
# 组件配置数据类
# =============================================================================

@dataclass
class ComponentConfig:
    """组件配置"""
    component_type: str  # retriever, graph_builder, textualizer
    component_name: str  # semantic, vector, keyword, pcst, simple, etc.
    config: Dict[str, Any]  # 具体配置参数

@dataclass
class ProcessorConfig:
    """处理器配置"""
    processor_name: str
    retriever_config: ComponentConfig
    graph_builder_config: ComponentConfig
    textualizer_config: ComponentConfig
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1小时
    max_tokens: int = 4000

# =============================================================================
# 组件工厂类
# =============================================================================

class ComponentFactory:
    """组件工厂"""
    
    def __init__(self):
        """初始化组件工厂"""
        self._retriever_registry: Dict[str, Type[BaseRetriever]] = {}
        self._graph_builder_registry: Dict[str, Type[BaseGraphBuilder]] = {}
        self._textualizer_registry: Dict[str, Type[BaseTextualizer]] = {}
        self._processor_registry: Dict[str, Type] = {}  # 添加处理器注册表
        
        # 注册默认组件
        self._register_default_components()
    
    def _register_default_components(self):
        """注册默认组件"""
        # 这里将在具体组件实现后注册
        logger.info("📝 组件工厂初始化，等待组件注册...")
    
    # =========================================================================
    # 组件注册方法
    # =========================================================================
    
    def register_retriever(self, name: str, retriever_class: Type[BaseRetriever]):
        """注册检索器"""
        self._retriever_registry[name] = retriever_class
        logger.info(f"✅ 注册检索器: {name}")
    
    def register_graph_builder(self, name: str, builder_class: Type[BaseGraphBuilder]):
        """注册图构建器"""
        self._graph_builder_registry[name] = builder_class
        logger.info(f"✅ 注册图构建器: {name}")
    
    def register_textualizer(self, name: str, textualizer_class: Type[BaseTextualizer]):
        """注册文本化器"""
        self._textualizer_registry[name] = textualizer_class
        logger.info(f"✅ 注册文本化器: {name}")
    
    def register_processor(self, name: str, processor_class: Type):
        """注册处理器"""
        self._processor_registry[name] = processor_class
        logger.info(f"✅ 注册处理器: {name}")
    
    # =========================================================================
    # 组件创建方法
    # =========================================================================
    
    def create_retriever(self, config: ComponentConfig) -> BaseRetriever:
        """创建检索器实例"""
        if config.component_name not in self._retriever_registry:
            raise ValueError(f"未找到检索器: {config.component_name}")
        
        retriever_class = self._retriever_registry[config.component_name]
        
        try:
            # 尝试使用增强配置（包含连接）
            enhanced_config = config.config.copy()
            enhanced_config['nebula_conn'] = nebula_conn
            retriever = retriever_class(**enhanced_config)
        except TypeError:
            # 如果构造函数不支持nebula_conn参数，使用原始配置
            logger.warning(f"检索器 {config.component_name} 不支持nebula_conn参数，使用原始配置")
            retriever = retriever_class(**config.config)
        
        logger.info(f"🔧 创建检索器: {config.component_name}")
        return retriever
    
    def create_graph_builder(self, config: ComponentConfig) -> BaseGraphBuilder:
        """创建图构建器实例"""
        if config.component_name not in self._graph_builder_registry:
            raise ValueError(f"未找到图构建器: {config.component_name}")
        
        builder_class = self._graph_builder_registry[config.component_name]
        
        try:
            # 尝试使用增强配置（包含连接）
            enhanced_config = config.config.copy()
            enhanced_config['nebula_conn'] = nebula_conn
            builder = builder_class(**enhanced_config)
        except TypeError:
            # 如果构造函数不支持nebula_conn参数，使用原始配置
            logger.warning(f"图构建器 {config.component_name} 不支持nebula_conn参数，使用原始配置")
            builder = builder_class(**config.config)
        
        logger.info(f"🔧 创建图构建器: {config.component_name}")
        return builder
    
    def create_textualizer(self, config: ComponentConfig) -> BaseTextualizer:
        """创建文本化器实例"""
        if config.component_name not in self._textualizer_registry:
            raise ValueError(f"未找到文本化器: {config.component_name}")
        
        textualizer_class = self._textualizer_registry[config.component_name]
        
        try:
            # 尝试使用增强配置（文本化器通常不需要连接）
            enhanced_config = config.config.copy()
            enhanced_config['nebula_conn'] = nebula_conn
            textualizer = textualizer_class(**enhanced_config)
        except TypeError:
            # 如果构造函数不支持nebula_conn参数，使用原始配置
            textualizer = textualizer_class(**config.config)
        
        logger.info(f"🔧 创建文本化器: {config.component_name}")
        return textualizer
        
        logger.info(f"🔧 创建文本化器: {config.component_name}")
        return textualizer
    
    # =========================================================================
    # 完整处理器组件创建
    # =========================================================================
    
    def create_processor_components(self, config: ProcessorConfig) -> Dict[str, Any]:
        """为处理器创建完整的组件集合"""
        try:
            logger.info(f"🏭 为处理器 {config.processor_name} 创建组件...")
            
            # 创建检索器
            retriever = self.create_retriever(config.retriever_config)
            
            # 创建图构建器
            graph_builder = self.create_graph_builder(config.graph_builder_config)
            
            # 创建文本化器
            textualizer = self.create_textualizer(config.textualizer_config)
            
            # 初始化所有组件
            if not retriever.initialize():
                raise RuntimeError(f"检索器 {config.retriever_config.component_name} 初始化失败")
            
            if not graph_builder.initialize():
                raise RuntimeError(f"图构建器 {config.graph_builder_config.component_name} 初始化失败")
            
            if not textualizer.initialize():
                raise RuntimeError(f"文本化器 {config.textualizer_config.component_name} 初始化失败")
            
            components = {
                'retriever': retriever,
                'graph_builder': graph_builder,
                'textualizer': textualizer,
                'config': config
            }
            
            logger.info(f"✅ 处理器 {config.processor_name} 组件创建完成")
            return components
            
        except Exception as e:
            logger.error(f"❌ 处理器 {config.processor_name} 组件创建失败: {str(e)}")
            raise
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    
    def list_available_components(self) -> Dict[str, List[str]]:
        """列出所有可用组件"""
        return {
            'retrievers': list(self._retriever_registry.keys()),
            'graph_builders': list(self._graph_builder_registry.keys()),
            'textualizers': list(self._textualizer_registry.keys())
        }
    
    def validate_config(self, config: ProcessorConfig) -> bool:
        """验证处理器配置"""
        try:
            # 检查检索器
            if config.retriever_config.component_name not in self._retriever_registry:
                logger.error(f"❌ 未找到检索器: {config.retriever_config.component_name}")
                return False
            
            # 检查图构建器
            if config.graph_builder_config.component_name not in self._graph_builder_registry:
                logger.error(f"❌ 未找到图构建器: {config.graph_builder_config.component_name}")
                return False
            
            # 检查文本化器
            if config.textualizer_config.component_name not in self._textualizer_registry:
                logger.error(f"❌ 未找到文本化器: {config.textualizer_config.component_name}")
                return False
            
            logger.info(f"✅ 处理器配置 {config.processor_name} 验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {str(e)}")
            return False

# =============================================================================
# 全局工厂实例
# =============================================================================

# 创建全局组件工厂实例
component_factory = ComponentFactory()

# =============================================================================
# 默认配置模板
# =============================================================================

class DefaultConfigs:
    """默认配置模板"""
    
    @staticmethod
    def get_semantic_retriever_config() -> ComponentConfig:
        """语义检索器配置"""
        return ComponentConfig(
            component_type='retriever',
            component_name='semantic',
            config={
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'top_k': 5,
                'similarity_threshold': 0.3
            }
        )
    
    @staticmethod
    def get_vector_retriever_config() -> ComponentConfig:
        """向量检索器配置"""
        return ComponentConfig(
            component_type='retriever',
            component_name='vector',
            config={
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'top_k': 5,
                'use_faiss': True
            }
        )
    
    @staticmethod
    def get_keyword_retriever_config() -> ComponentConfig:
        """关键词检索器配置"""
        return ComponentConfig(
            component_type='retriever',
            component_name='keyword',
            config={
                'top_k': 5,
                'use_tfidf': True,
                'min_score': 0.1
            }
        )
    
    @staticmethod
    def get_pcst_graph_builder_config() -> ComponentConfig:
        """PCST图构建器配置"""
        return ComponentConfig(
            component_type='graph_builder',
            component_name='pcst',
            config={
                'max_nodes': 20,
                'prize_weight': 1.0,
                'cost_weight': 0.5
            }
        )
    
    @staticmethod
    def get_simple_graph_builder_config() -> ComponentConfig:
        """简单图构建器配置"""
        return ComponentConfig(
            component_type='graph_builder',
            component_name='simple',
            config={
                'max_nodes': 15,
                'max_depth': 2
            }
        )
    
    @staticmethod
    def get_template_textualizer_config() -> ComponentConfig:
        """模板文本化器配置"""
        return ComponentConfig(
            component_type='textualizer',
            component_name='template',
            config={
                'template_type': 'qa',
                'include_properties': True,
                'max_tokens': 2000
            }
        )
    
    @staticmethod
    def get_compact_textualizer_config() -> ComponentConfig:
        """紧凑文本化器配置"""
        return ComponentConfig(
            component_type='textualizer',
            component_name='compact',
            config={
                'max_tokens': 1000,
                'prioritize_entities': True
            }
        )
    
    @staticmethod
    def get_qa_textualizer_config() -> ComponentConfig:
        """QA文本化器配置"""
        return ComponentConfig(
            component_type='textualizer',
            component_name='qa',
            config={
                'focus_on_query': True,
                'max_tokens': 1500
            }
        )
    
    @staticmethod
    def get_gnn_graph_builder_config() -> ComponentConfig:
        """GNN图构建器配置"""
        return ComponentConfig(
            component_type='graph_builder',
            component_name='gnn',
            config={
                'max_nodes': 50,
                'max_hops': 2,
                'feature_dim': 768,
                'include_edge_features': True
            }
        )

# =============================================================================
# 处理器默认配置
# =============================================================================

class ProcessorDefaultConfigs:
    """处理器默认配置"""
    
    @staticmethod
    def get_direct_processor_config() -> ProcessorConfig:
        """直接查询处理器配置"""
        return ProcessorConfig(
            processor_name='direct',
            retriever_config=DefaultConfigs.get_keyword_retriever_config(),
            graph_builder_config=DefaultConfigs.get_simple_graph_builder_config(),
            textualizer_config=DefaultConfigs.get_qa_textualizer_config(),  # 使用QA文本化器以支持属性信息
            cache_enabled=True,
            cache_ttl=7200,  # 2小时
            max_tokens=2000
        )
    
    @staticmethod
    def get_simple_g_processor_config() -> ProcessorConfig:
        """简单图查询处理器配置"""
        return ProcessorConfig(
            processor_name='simple_g',
            retriever_config=DefaultConfigs.get_semantic_retriever_config(),
            graph_builder_config=DefaultConfigs.get_simple_graph_builder_config(),
            textualizer_config=DefaultConfigs.get_template_textualizer_config(),
            cache_enabled=True,
            cache_ttl=3600,  # 1小时
            max_tokens=3000
        )
    
    @staticmethod
    def get_complex_g_processor_config() -> ProcessorConfig:
        """复杂图查询处理器配置"""
        return ProcessorConfig(
            processor_name='complex_g',
            retriever_config=DefaultConfigs.get_semantic_retriever_config(),
            graph_builder_config=DefaultConfigs.get_pcst_graph_builder_config(),
            textualizer_config=DefaultConfigs.get_template_textualizer_config(),
            cache_enabled=True,
            cache_ttl=1800,  # 30分钟
            max_tokens=4000
        )
    
    @staticmethod
    def get_comparison_processor_config() -> ProcessorConfig:
        """比较查询处理器配置"""
        return ProcessorConfig(
            processor_name='comparison',
            retriever_config=DefaultConfigs.get_vector_retriever_config(),
            graph_builder_config=DefaultConfigs.get_pcst_graph_builder_config(),
            textualizer_config=DefaultConfigs.get_template_textualizer_config(),
            cache_enabled=True,
            cache_ttl=3600,  # 1小时
            max_tokens=3500
        )
    
    @staticmethod
    def get_chitchat_processor_config() -> ProcessorConfig:
        """闲聊处理器配置"""
        return ProcessorConfig(
            processor_name='chitchat',
            retriever_config=DefaultConfigs.get_keyword_retriever_config(),
            graph_builder_config=DefaultConfigs.get_simple_graph_builder_config(),
            textualizer_config=DefaultConfigs.get_compact_textualizer_config(),
            cache_enabled=False,  # 闲聊不需要缓存
            cache_ttl=0,
            max_tokens=1500
        )
    
    @staticmethod
    def get_gnn_processor_config() -> ProcessorConfig:
        """GNN处理器配置"""
        return ProcessorConfig(
            processor_name='gnn',
            retriever_config=DefaultConfigs.get_semantic_retriever_config(),
            graph_builder_config=ComponentConfig(
                component_type='graph_builder',
                component_name='gnn',
                config={
                    'max_nodes': 50,
                    'max_hops': 2,
                    'feature_dim': 768,
                    'include_edge_features': True
                }
            ),
            textualizer_config=DefaultConfigs.get_template_textualizer_config(),
            cache_enabled=True,
            cache_ttl=1800,  # 30分钟
            max_tokens=4500
        )
