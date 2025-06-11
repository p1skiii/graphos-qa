"""
Prompt模板管理系统
为不同类型的查询和处理器提供适配的prompt模板
支持Phi-3-mini的格式要求
"""
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import json

from .input_router import UnifiedInput

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Prompt类型枚举"""
    SIMPLE_QA = "simple_qa"
    MULTIMODAL_QA = "multimodal_qa"
    COMPARISON_QA = "comparison_qa"
    CHITCHAT = "chitchat"
    SYSTEM = "system"

@dataclass
class PromptTemplate:
    """Prompt模板数据结构"""
    name: str
    template: str
    prompt_type: PromptType
    required_fields: List[str]
    optional_fields: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = []
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        try:
            # 检查必需字段
            missing_fields = [field for field in self.required_fields if field not in kwargs]
            if missing_fields:
                raise ValueError(f"缺少必需字段: {missing_fields}")
            
            # 填充默认值
            format_kwargs = kwargs.copy()
            for field in self.optional_fields:
                if field not in format_kwargs:
                    format_kwargs[field] = ""
            
            return self.template.format(**format_kwargs)
            
        except KeyError as e:
            raise ValueError(f"模板格式化失败，缺少字段: {str(e)}")
    
    def validate(self) -> bool:
        """验证模板"""
        try:
            # 创建测试参数
            test_kwargs = {}
            for field in self.required_fields:
                test_kwargs[field] = f"test_{field}"
            for field in self.optional_fields:
                test_kwargs[field] = f"test_{field}"
            
            # 尝试格式化
            self.format(**test_kwargs)
            return True
            
        except Exception as e:
            logger.error(f"模板验证失败: {str(e)}")
            return False

class PromptTemplateManager:
    """Prompt模板管理器"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates: Dict[str, PromptTemplate] = {}
        self.processor_mappings: Dict[str, str] = {
            'direct': 'simple_qa',
            'simple_g': 'simple_qa', 
            'complex_g': 'multimodal_qa',
            'comparison': 'comparison_qa',
            'chitchat': 'chitchat'
        }
        
        # 初始化默认模板
        self._init_default_templates()
        
        logger.info(f"📝 PromptTemplateManager初始化完成，加载了{len(self.templates)}个模板")
    
    def _init_default_templates(self):
        """初始化默认模板"""
        
        # 系统Prompt模板
        system_template = PromptTemplate(
            name="system_basketball_qa",
            template="""{system_message}

请遵循以下原则：
1. 基于提供的上下文信息准确回答问题
2. 如果信息不足，请明确说明
3. 保持回答简洁且专业
4. 只回答篮球相关的问题""",
            prompt_type=PromptType.SYSTEM,
            required_fields=["system_message"],
            description="篮球问答系统Prompt"
        )
        
        # 简单问答模板
        simple_qa_template = PromptTemplate(
            name="simple_qa",
            template="""<|system|>
{system_prompt}<|end|>
<|user|>
基于以下上下文信息回答问题：

上下文：
{context}

问题：{query}

请基于上下文提供准确的答案。<|end|>
<|assistant|>
""",
            prompt_type=PromptType.SIMPLE_QA,
            required_fields=["system_prompt", "context", "query"],
            description="简单问答模板（Direct/Simple G处理器）"
        )
        
        # 多模态问答模板
        multimodal_qa_template = PromptTemplate(
            name="multimodal_qa",
            template="""<|system|>
{system_prompt}<|end|>
<|user|>
基于以下文本信息和图谱分析回答问题：

文本上下文：
{text_context}

图谱信息：
{graph_info}

问题：{query}

请综合文本和图谱信息提供全面的答案。<|end|>
<|assistant|>
""",
            prompt_type=PromptType.MULTIMODAL_QA,
            required_fields=["system_prompt", "text_context", "graph_info", "query"],
            description="多模态问答模板（Complex G处理器增强模式）"
        )
        
        # 比较问答模板
        comparison_qa_template = PromptTemplate(
            name="comparison_qa",
            template="""<|system|>
{system_prompt}<|end|>
<|user|>
基于以下信息进行比较分析：

比较对象：{comparison_subjects}
比较维度：{comparison_aspects}

相关信息：
{context}

问题：{query}

请提供详细的比较分析。<|end|>
<|assistant|>
""",
            prompt_type=PromptType.COMPARISON_QA,
            required_fields=["system_prompt", "comparison_subjects", "comparison_aspects", "context", "query"],
            optional_fields=["additional_context"],
            description="比较问答模板（Comparison处理器）"
        )
        
        # 闲聊模板
        chitchat_template = PromptTemplate(
            name="chitchat",
            template="""<|system|>
{system_prompt}

你现在处于闲聊模式，可以进行轻松的篮球话题讨论。<|end|>
<|user|>
{query}<|end|>
<|assistant|>
""",
            prompt_type=PromptType.CHITCHAT,
            required_fields=["system_prompt", "query"],
            optional_fields=["context"],
            description="闲聊模板（Chitchat处理器）"
        )
        
        # 注册所有模板
        templates = [
            system_template,
            simple_qa_template, 
            multimodal_qa_template,
            comparison_qa_template,
            chitchat_template
        ]
        
        for template in templates:
            self.register_template(template)
    
    def register_template(self, template: PromptTemplate) -> bool:
        """注册模板"""
        try:
            # 验证模板
            if not template.validate():
                raise ValueError(f"模板验证失败: {template.name}")
            
            self.templates[template.name] = template
            logger.info(f"✅ 注册模板: {template.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模板注册失败: {str(e)}")
            return False
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(template_name)
    
    def get_template_for_processor(self, processor_type: str) -> Optional[PromptTemplate]:
        """根据处理器类型获取模板"""
        template_name = self.processor_mappings.get(processor_type)
        if template_name:
            return self.get_template(template_name)
        return None
    
    def format_prompt_for_input(self, unified_input: UnifiedInput, system_prompt: str = None) -> str:
        """为统一输入格式化prompt"""
        try:
            # 获取适当的模板
            template = self.get_template_for_processor(unified_input.processor_type)
            if not template:
                # 使用默认简单QA模板
                template = self.get_template('simple_qa')
            
            if not template:
                raise ValueError("找不到合适的模板")
            
            # 准备格式化参数
            format_kwargs = self._prepare_format_kwargs(unified_input, template, system_prompt)
            
            # 格式化prompt
            prompt = template.format(**format_kwargs)
            
            logger.info(f"📝 为{unified_input.processor_type}处理器生成prompt，长度: {len(prompt)}")
            return prompt
            
        except Exception as e:
            logger.error(f"❌ Prompt格式化失败: {str(e)}")
            # 返回简单的回退prompt
            return self._create_fallback_prompt(unified_input, system_prompt)
    
    def _prepare_format_kwargs(self, unified_input: UnifiedInput, template: PromptTemplate, system_prompt: str = None) -> Dict[str, Any]:
        """准备格式化参数"""
        # 默认系统prompt
        default_system_prompt = system_prompt or "你是一个专业的篮球知识问答助手。"
        
        kwargs = {
            'system_prompt': default_system_prompt,
            'query': unified_input.query
        }
        
        # 根据模板类型准备不同的参数
        if template.prompt_type == PromptType.SIMPLE_QA:
            kwargs.update({
                'context': unified_input.get_text_content() or "暂无相关信息"
            })
            
        elif template.prompt_type == PromptType.MULTIMODAL_QA:
            text_context = unified_input.get_text_content() or "暂无文本信息"
            graph_info = self._format_graph_info(unified_input)
            
            kwargs.update({
                'text_context': text_context,
                'graph_info': graph_info
            })
            
        elif template.prompt_type == PromptType.COMPARISON_QA:
            # 从processor_data中提取比较信息
            processor_data = unified_input.processor_data or {}
            kwargs.update({
                'comparison_subjects': processor_data.get('comparison_subjects', '未知对象'),
                'comparison_aspects': processor_data.get('comparison_aspects', '多个方面'),
                'context': unified_input.get_text_content() or "暂无相关信息"
            })
            
        elif template.prompt_type == PromptType.CHITCHAT:
            # 闲聊模式可选上下文
            context = unified_input.get_text_content()
            if context:
                kwargs['context'] = context
        
        return kwargs
    
    def _format_graph_info(self, unified_input: UnifiedInput) -> str:
        """格式化图信息"""
        if not unified_input.has_multimodal_data():
            return "暂无图谱信息"
        
        graph_embedding = unified_input.get_graph_embedding()
        if graph_embedding:
            embedding_summary = f"图嵌入向量（{len(graph_embedding)}维）"
            
            # 如果有多模态上下文，尝试提取更多信息
            if unified_input.multimodal_context:
                metadata = unified_input.multimodal_context.metadata
                if metadata:
                    embedding_summary += f"，创建时间: {metadata.get('creation_time', '未知')}"
            
            return f"图谱分析结果：{embedding_summary}"
        
        return "图谱信息处理中"
    
    def _create_fallback_prompt(self, unified_input: UnifiedInput, system_prompt: str = None) -> str:
        """创建回退prompt"""
        system_msg = system_prompt or "你是一个专业的篮球知识问答助手。"
        context = unified_input.get_text_content() or "暂无相关信息"
        
        return f"""<|system|>
{system_msg}<|end|>
<|user|>
基于以下信息回答问题：

{context}

问题：{unified_input.query}<|end|>
<|assistant|>
"""
    
    def list_templates(self) -> List[str]:
        """列出所有模板名称"""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """获取模板信息"""
        template = self.get_template(template_name)
        if template:
            return {
                'name': template.name,
                'type': template.prompt_type.value,
                'required_fields': template.required_fields,
                'optional_fields': template.optional_fields,
                'description': template.description,
                'is_valid': template.validate()
            }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        template_count_by_type = {}
        for template in self.templates.values():
            type_name = template.prompt_type.value
            template_count_by_type[type_name] = template_count_by_type.get(type_name, 0) + 1
        
        return {
            'total_templates': len(self.templates),
            'templates_by_type': template_count_by_type,
            'processor_mappings': self.processor_mappings.copy(),
            'available_templates': self.list_templates()
        }

# =============================================================================
# 工厂函数
# =============================================================================

def create_prompt_template_manager() -> PromptTemplateManager:
    """创建prompt模板管理器"""
    return PromptTemplateManager()

# =============================================================================
# 测试和演示
# =============================================================================

if __name__ == "__main__":
    # 创建模板管理器
    manager = create_prompt_template_manager()
    
    print("📝 Prompt模板管理器测试")
    print(f"总模板数: {len(manager.templates)}")
    
    # 测试每种模板
    test_cases = [
        {
            'template_name': 'simple_qa',
            'kwargs': {
                'system_prompt': '你是篮球专家',
                'context': '科比是洛杉矶湖人队的传奇球员',
                'query': '科比是谁？'
            }
        },
        {
            'template_name': 'multimodal_qa',
            'kwargs': {
                'system_prompt': '你是篮球专家',
                'text_context': '科比和詹姆斯都是NBA巨星',
                'graph_info': '图谱显示他们有队友关系',
                'query': '科比和詹姆斯什么关系？'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 测试模板 {i+1}: {test_case['template_name']}")
        template = manager.get_template(test_case['template_name'])
        if template:
            try:
                prompt = template.format(**test_case['kwargs'])
                print(f"✅ 模板格式化成功，长度: {len(prompt)}")
                print(f"预览: {prompt[:100]}...")
            except Exception as e:
                print(f"❌ 模板格式化失败: {str(e)}")
        else:
            print(f"❌ 模板不存在")
    
    # 显示统计信息
    print(f"\n📊 管理器统计: {manager.get_stats()}")
