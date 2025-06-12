"""
LLM工厂类
提供统一的LLM组件创建和管理接口
支持不同配置和使用场景
"""
import logging
from typing import Dict, Any, Optional, List, Type, Union
from dataclasses import dataclass

from .config import LLMConfig, Phi3Config, get_macos_optimized_config, get_default_llm_config
from .llm_engine import LLMEngine, create_llm_engine
from .input_router import InputRouter, LLMInputConfig, create_input_router
from .prompt_templates import PromptTemplateManager, create_prompt_template_manager
from .response_formatter import ResponseFormatter, create_response_formatter

logger = logging.getLogger(__name__)

@dataclass
class LLMSystemConfig:
    """LLM系统总配置"""
    llm_config: LLMConfig
    input_config: LLMInputConfig
    formatter_config: Dict[str, Any]
    auto_load_model: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'llm_config': self.llm_config.to_dict(),
            'input_config': {
                'processor_type': self.input_config.processor_type,
                'enable_multimodal': self.input_config.enable_multimodal,
                'max_tokens': self.input_config.max_tokens,
                'include_metadata': self.input_config.include_metadata
            },
            'formatter_config': self.formatter_config,
            'auto_load_model': self.auto_load_model
        }

class LLMSystem:
    """完整的LLM系统"""
    
    def __init__(self, config: LLMSystemConfig):
        """初始化LLM系统"""
        self.config = config
        
        # 核心组件
        self.engine: Optional[LLMEngine] = None
        self.input_router: Optional[InputRouter] = None
        self.prompt_manager: Optional[PromptTemplateManager] = None
        self.response_formatter: Optional[ResponseFormatter] = None
        
        # 状态
        self.is_initialized = False
        self.is_ready = False
        
        logger.info("🏗️ LLMSystem初始化开始")
    
    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("🔄 初始化LLM系统组件...")
            
            # 1. 创建输入路由器
            self.input_router = create_input_router(self.config.input_config)
            logger.info("✅ 输入路由器创建完成")
            
            # 2. 创建prompt管理器
            self.prompt_manager = create_prompt_template_manager()
            logger.info("✅ Prompt管理器创建完成")
            
            # 3. 创建响应格式化器
            self.response_formatter = create_response_formatter(self.config.formatter_config)
            logger.info("✅ 响应格式化器创建完成")
            
            # 4. 创建LLM引擎
            self.engine = create_llm_engine(self.config.llm_config)
            logger.info("✅ LLM引擎创建完成")
            
            self.is_initialized = True
            
            # 5. 自动加载模型（如果配置）
            if self.config.auto_load_model:
                logger.info("🔄 自动加载模型...")
                if self.engine.load_model():
                    self.is_ready = True
                    logger.info("✅ 模型加载完成，系统就绪")
                else:
                    logger.warning("⚠️ 模型加载失败，系统未就绪")
            else:
                logger.info("ℹ️ 模型需要手动加载")
            
            logger.info("🎉 LLM系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ LLM系统初始化失败: {str(e)}")
            return False
    
    def load_model(self) -> bool:
        """加载模型"""
        if not self.is_initialized:
            logger.error("❌ 系统未初始化，无法加载模型")
            return False
        
        if not self.engine:
            logger.error("❌ LLM引擎未创建")
            return False
        
        try:
            logger.info("🔄 开始加载LLM模型...")
            success = self.engine.load_model()
            
            if success:
                self.is_ready = True
                logger.info("✅ 模型加载成功，系统就绪")
            else:
                logger.error("❌ 模型加载失败")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 模型加载异常: {str(e)}")
            return False
    
    def process_query(self, query: str, processor_output: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询"""
        if not self.is_ready:
            return {
                'success': False,
                'error': '系统未就绪，请先加载模型',
                'content': 'Sorry, the system is initializing, please try again later.'
            }
        
        try:
            # 1. 输入路由
            unified_input = self.input_router.route_processor_output(processor_output, query)
            
            # 2. LLM推理
            llm_response = self.engine.generate_response(unified_input)
            
            # 3. 响应格式化
            formatted_response = self.response_formatter.format_response(llm_response, unified_input)
            
            # 4. 构建最终响应
            final_response = {
                'success': True,
                'content': formatted_response.content,
                'metadata': formatted_response.metadata,
                'format_type': formatted_response.format_type,
                'processing_info': formatted_response.processing_info,
                'system_info': {
                    'model_name': self.config.llm_config.model_config.model_name,
                    'processor_type': unified_input.processor_type,
                    'multimodal': unified_input.has_multimodal_data()
                }
            }
            
            logger.info(f"✅ 查询处理完成: {query[:50]}...")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ 查询处理失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': f'Sorry, an error occurred while processing your request: {str(e)}'
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'initialized': self.is_initialized,
            'ready': self.is_ready,
            'components': {
                'engine': self.engine is not None,
                'input_router': self.input_router is not None,
                'prompt_manager': self.prompt_manager is not None,
                'response_formatter': self.response_formatter is not None
            }
        }
        
        # 如果组件存在，添加详细状态
        if self.engine:
            status['engine_stats'] = self.engine.get_stats()
        
        if self.input_router:
            status['router_stats'] = self.input_router.get_stats()
        
        if self.response_formatter:
            status['formatter_stats'] = self.response_formatter.get_stats()
        
        return status

class LLMFactory:
    """LLM工厂类"""
    
    def __init__(self):
        """初始化工厂"""
        self.presets = {
            'macos_optimized': self._create_macos_optimized_config,
            'default': self._create_default_config,
            'production': self._create_production_config,
            'development': self._create_development_config
        }
        
        logger.info(f"🏭 LLMFactory初始化完成，可用预设: {list(self.presets.keys())}")
    
    def create_system(self, preset: str = 'macos_optimized', custom_config: Optional[Dict[str, Any]] = None) -> LLMSystem:
        """创建LLM系统
        
        Args:
            preset: 预设配置名称
            custom_config: 自定义配置覆盖
            
        Returns:
            LLMSystem: 配置好的LLM系统
        """
        try:
            # 获取预设配置
            if preset in self.presets:
                config = self.presets[preset]()
            else:
                logger.warning(f"⚠️ 未知预设: {preset}，使用默认配置")
                config = self._create_default_config()
            
            # 应用自定义配置
            if custom_config:
                config = self._apply_custom_config(config, custom_config)
            
            # 创建系统
            system = LLMSystem(config)
            
            logger.info(f"✅ LLM系统创建完成，预设: {preset}")
            return system
            
        except Exception as e:
            logger.error(f"❌ LLM系统创建失败: {str(e)}")
            raise
    
    def _create_macos_optimized_config(self) -> LLMSystemConfig:
        """创建MacOS优化配置"""
        llm_config = get_macos_optimized_config()
        
        input_config = LLMInputConfig(
            processor_type='auto',
            enable_multimodal=True,
            max_tokens=3072,
            include_metadata=True
        )
        
        formatter_config = {
            'enable_markdown': True,
            'add_metadata': True,
            'clean_output': True,
            'max_response_length': 2048,
            'format_lists': True,
            'highlight_keywords': False
        }
        
        return LLMSystemConfig(
            llm_config=llm_config,
            input_config=input_config,
            formatter_config=formatter_config,
            auto_load_model=False  # MacOS环境手动加载
        )
    
    def _create_default_config(self) -> LLMSystemConfig:
        """创建默认配置"""
        llm_config = get_default_llm_config()
        
        input_config = LLMInputConfig(
            processor_type='auto',
            enable_multimodal=True,
            max_tokens=3072,
            include_metadata=True
        )
        
        formatter_config = {
            'enable_markdown': True,
            'add_metadata': False,
            'clean_output': True,
            'max_response_length': 1024,
            'format_lists': True
        }
        
        return LLMSystemConfig(
            llm_config=llm_config,
            input_config=input_config,
            formatter_config=formatter_config,
            auto_load_model=False
        )
    
    def _create_production_config(self) -> LLMSystemConfig:
        """创建生产环境配置"""
        # 生产环境使用更严格的配置
        llm_config = get_default_llm_config()
        llm_config.enable_cache = True
        llm_config.cache_ttl = 7200  # 2小时缓存
        
        input_config = LLMInputConfig(
            processor_type='auto',
            enable_multimodal=True,
            max_tokens=3072,
            include_metadata=False  # 生产环境减少元数据
        )
        
        formatter_config = {
            'enable_markdown': True,
            'add_metadata': False,
            'clean_output': True,
            'max_response_length': 1536,
            'format_lists': True,
            'highlight_keywords': False
        }
        
        return LLMSystemConfig(
            llm_config=llm_config,
            input_config=input_config,
            formatter_config=formatter_config,
            auto_load_model=True  # 生产环境自动加载
        )
    
    def _create_development_config(self) -> LLMSystemConfig:
        """创建开发环境配置"""
        llm_config = get_macos_optimized_config()
        llm_config.enable_cache = False  # 开发环境不缓存
        
        input_config = LLMInputConfig(
            processor_type='auto',
            enable_multimodal=True,
            max_tokens=2048,  # 开发环境较小
            include_metadata=True
        )
        
        formatter_config = {
            'enable_markdown': True,
            'add_metadata': True,
            'clean_output': False,  # 开发环境保留原始输出
            'max_response_length': 1024,
            'format_lists': True,
            'highlight_keywords': True
        }
        
        return LLMSystemConfig(
            llm_config=llm_config,
            input_config=input_config,
            formatter_config=formatter_config,
            auto_load_model=False
        )
    
    def _apply_custom_config(self, base_config: LLMSystemConfig, custom_config: Dict[str, Any]) -> LLMSystemConfig:
        """应用自定义配置"""
        # 这里可以实现更复杂的配置合并逻辑
        # 简单实现：直接覆盖指定字段
        
        if 'llm_config' in custom_config:
            llm_custom = custom_config['llm_config']
            for key, value in llm_custom.items():
                if hasattr(base_config.llm_config, key):
                    setattr(base_config.llm_config, key, value)
        
        if 'input_config' in custom_config:
            input_custom = custom_config['input_config']
            for key, value in input_custom.items():
                if hasattr(base_config.input_config, key):
                    setattr(base_config.input_config, key, value)
        
        if 'formatter_config' in custom_config:
            base_config.formatter_config.update(custom_config['formatter_config'])
        
        if 'auto_load_model' in custom_config:
            base_config.auto_load_model = custom_config['auto_load_model']
        
        return base_config
    
    def list_presets(self) -> List[str]:
        """列出可用预设"""
        return list(self.presets.keys())
    
    def get_preset_info(self, preset: str) -> Optional[Dict[str, Any]]:
        """获取预设信息"""
        if preset not in self.presets:
            return None
        
        try:
            config = self.presets[preset]()
            return {
                'name': preset,
                'model_name': config.llm_config.model_config.model_name,
                'device': config.llm_config.model_config.device,
                'multimodal_enabled': config.llm_config.enable_multimodal,
                'auto_load_model': config.auto_load_model,
                'description': self._get_preset_description(preset)
            }
        except Exception as e:
            logger.error(f"❌ 获取预设信息失败: {str(e)}")
            return None
    
    def _get_preset_description(self, preset: str) -> str:
        """获取预设描述"""
        descriptions = {
            'macos_optimized': 'MacOS环境优化配置，使用CPU推理，适合本地开发',
            'default': '默认配置，平衡性能和资源使用',
            'production': '生产环境配置，启用缓存和自动加载',
            'development': '开发环境配置，详细日志和调试信息'
        }
        return descriptions.get(preset, '自定义配置')

# =============================================================================
# 全局工厂实例
# =============================================================================

llm_factory = LLMFactory()

# =============================================================================
# 便捷函数
# =============================================================================

def create_llm_system(preset: str = 'macos_optimized', custom_config: Optional[Dict[str, Any]] = None) -> LLMSystem:
    """创建LLM系统的便捷函数"""
    return llm_factory.create_system(preset, custom_config)

def get_available_presets() -> List[str]:
    """获取可用预设列表"""
    return llm_factory.list_presets()

# =============================================================================
# 测试和演示
# =============================================================================

if __name__ == "__main__":
    print("🏭 LLM工厂测试")
    
    # 显示可用预设
    presets = get_available_presets()
    print(f"可用预设: {presets}")
    
    # 获取每个预设的信息
    for preset in presets:
        info = llm_factory.get_preset_info(preset)
        if info:
            print(f"\n📋 {preset}:")
            print(f"   模型: {info['model_name']}")
            print(f"   设备: {info['device']}")
            print(f"   多模态: {info['multimodal_enabled']}")
            print(f"   描述: {info['description']}")
    
    # 创建测试系统
    print(f"\n🧪 创建测试系统...")
    system = create_llm_system('macos_optimized')
    
    print(f"✅ 系统创建完成")
    print(f"初始化状态: {system.is_initialized}")
    
    # 初始化系统
    if system.initialize():
        print(f"✅ 系统初始化成功")
        status = system.get_system_status()
        print(f"系统状态: 初始化={status['initialized']}, 就绪={status['ready']}")
    else:
        print(f"❌ 系统初始化失败")
