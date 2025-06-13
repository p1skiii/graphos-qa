"""
响应格式化器
统一处理LLM输出，支持不同格式和后处理需求
"""
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json
import re

from .llm_engine import LLMResponse
from .input_router import UnifiedInput

logger = logging.getLogger(__name__)

@dataclass
class FormattedResponse:
    """格式化后的响应"""
    content: str
    metadata: Dict[str, Any]
    format_type: str
    processing_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'format_type': self.format_type,
            'processing_info': self.processing_info
        }

class ResponseFormatter:
    """响应格式化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化响应格式化器"""
        self.config = config or {}
        
        # 默认配置
        self.default_config = {
            'enable_markdown': True,
            'add_metadata': True,
            'clean_output': True,
            'max_response_length': 2048,
            'add_citations': False,
            'format_lists': True,
            'highlight_keywords': False
        }
        
        # 合并配置
        self.settings = {**self.default_config, **self.config}
        
        # 格式化统计
        self.stats = {
            'total_formatted': 0,
            'by_processor_type': {},
            'formatting_time': 0.0,
            'truncated_responses': 0,
            'error_responses': 0
        }
        
        # 关键词高亮配置
        self.basketball_keywords = [
            '科比', '詹姆斯', '乔丹', '库里', '杜兰特', '字母哥',
            'NBA', '湖人', '勇士', '公牛', '凯尔特人', '热火',
            '篮球', '得分', '助攻', '篮板', '总冠军', '季后赛',
            '常规赛', '全明星', 'MVP', 'FMVP'
        ]
        
        logger.info(f"📝 ResponseFormatter初始化完成，配置: {self.settings}")
    
    def format_response(self, llm_response: LLMResponse, unified_input: UnifiedInput) -> FormattedResponse:
        """格式化响应"""
        start_time = time.time()
        
        try:
            # 更新统计
            self.stats['total_formatted'] += 1
            processor_type = unified_input.processor_type
            if processor_type not in self.stats['by_processor_type']:
                self.stats['by_processor_type'][processor_type] = 0
            self.stats['by_processor_type'][processor_type] += 1
            
            # 基础清理
            content = self._clean_content(llm_response.content)
            
            # 长度检查和截断
            if len(content) > self.settings['max_response_length']:
                content = self._truncate_content(content)
                self.stats['truncated_responses'] += 1
            
            # 根据处理器类型选择格式化策略
            content = self._format_by_processor_type(content, unified_input)
            
            # Markdown格式化
            if self.settings['enable_markdown']:
                content = self._format_markdown(content)
            
            # 列表格式化
            if self.settings['format_lists']:
                content = self._format_lists(content)
            
            # 关键词高亮
            if self.settings['highlight_keywords']:
                content = self._highlight_keywords(content)
            
            # 构建元数据
            metadata = self._build_metadata(llm_response, unified_input)
            
            # 处理信息
            processing_info = self._build_processing_info(llm_response, start_time)
            
            # 确定格式类型
            format_type = self._determine_format_type(unified_input, content)
            
            formatting_time = time.time() - start_time
            self.stats['formatting_time'] += formatting_time
            
            logger.info(f"✅ 响应格式化完成，处理器: {processor_type}，耗时: {formatting_time:.3f}秒")
            
            return FormattedResponse(
                content=content,
                metadata=metadata,
                format_type=format_type,
                processing_info=processing_info
            )
            
        except Exception as e:
            logger.error(f"❌ 响应格式化失败: {str(e)}")
            self.stats['error_responses'] += 1
            return self._create_error_formatted_response(llm_response, unified_input, str(e))
    
    def _clean_content(self, content: str) -> str:
        """清理内容"""
        if not self.settings['clean_output']:
            return content
        
        # 移除多余的空白字符
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # 多个连续换行
        content = re.sub(r' +', ' ', content)  # 多个连续空格
        content = content.strip()
        
        # 移除可能的模型生成的特殊标记
        content = re.sub(r'<\|.*?\|>', '', content)
        
        # 移除重复的句子（简单检测）
        sentences = content.split('。')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        if len(unique_sentences) < len(sentences):
            content = '。'.join(unique_sentences)
            if content and not content.endswith('。'):
                content += '。'
        
        return content
    
    def _truncate_content(self, content: str) -> str:
        """截断内容"""
        max_length = self.settings['max_response_length']
        
        if len(content) <= max_length:
            return content
        
        # 尝试在句号处截断
        truncated = content[:max_length]
        last_period = truncated.rfind('。')
        
        if last_period > max_length * 0.7:  # 如果句号位置合理
            truncated = truncated[:last_period + 1]
        else:
            # 在空格处截断
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                truncated = truncated[:last_space]
        
        # 添加省略号
        if not truncated.endswith('。'):
            truncated += '...'
        
        logger.warning(f"⚠️ 响应被截断，原长度: {len(content)}，截断后: {len(truncated)}")
        return truncated
    
    def _format_by_processor_type(self, content: str, unified_input: UnifiedInput) -> str:
        """根据处理器类型格式化"""
        processor_type = unified_input.processor_type
        
        if processor_type == 'comparison':
            return self._format_comparison_response(content)
        elif processor_type == 'chitchat':
            return self._format_chitchat_response(content)
        elif processor_type == 'complex_g':
            return self._format_complex_response(content, unified_input)
        else:
            return self._format_simple_response(content)
    
    def _format_comparison_response(self, content: str) -> str:
        """格式化比较类响应"""
        # 添加比较结构
        if '比较' in content and not content.startswith('## 比较分析'):
            content = f"## 比较分析\n\n{content}"
        
        # 识别并格式化对比点
        content = re.sub(r'(\d+\.|[一二三四五六七八九十]+、)', r'\n**\1**', content)
        
        return content
    
    def _format_chitchat_response(self, content: str) -> str:
        """格式化闲聊类响应"""
        # 使用更轻松的语调
        if not any(emoji in content for emoji in ['😊', '👍', '🏀']):
            if '篮球' in content:
                content += ' 🏀'
        
        return content
    
    def _format_complex_response(self, content: str, unified_input: UnifiedInput) -> str:
        """格式化复杂查询响应"""
        # 如果有多模态数据，添加说明
        if unified_input.has_multimodal_data():
            if not content.startswith('基于图谱分析'):
                content = f"基于图谱分析和文本信息：\n\n{content}"
        
        return content
    
    def _format_simple_response(self, content: str) -> str:
        """格式化简单响应"""
        return content
    
    def _format_markdown(self, content: str) -> str:
        """Markdown格式化"""
        # 自动识别标题
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # 检测可能的标题
            if (len(line) < 50 and 
                ('分析' in line or '总结' in line or '结论' in line) and 
                line.endswith(('：', ':'))):
                formatted_lines.append(f"### {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_lists(self, content: str) -> str:
        """格式化列表"""
        # 识别并格式化数字列表
        content = re.sub(r'^(\d+)\.\s*', r'- **\1.** ', content, flags=re.MULTILINE)
        
        # 识别并格式化中文数字列表
        chinese_numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        for i, num in enumerate(chinese_numbers):
            pattern = f'^{num}、\\s*'
            replacement = f'- **{i+1}.** '
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def _highlight_keywords(self, content: str) -> str:
        """高亮关键词"""
        for keyword in self.basketball_keywords:
            if keyword in content:
                # 使用Markdown粗体
                content = content.replace(keyword, f"**{keyword}**")
        
        return content
    
    def _build_metadata(self, llm_response: LLMResponse, unified_input: UnifiedInput) -> Dict[str, Any]:
        """构建元数据"""
        metadata = llm_response.metadata.copy()
        
        if self.settings['add_metadata']:
            metadata.update({
                'query': unified_input.query,
                'processor_type': unified_input.processor_type,
                'response_length': len(llm_response.content),
                'has_multimodal_data': unified_input.has_multimodal_data(),
                'formatting_applied': True,
                'formatting_config': self.settings
            })
        
        return metadata
    
    def _build_processing_info(self, llm_response: LLMResponse, start_time: float) -> Dict[str, Any]:
        """构建处理信息"""
        formatting_time = time.time() - start_time
        
        return {
            'llm_processing_time': llm_response.processing_time,
            'formatting_time': formatting_time,
            'total_time': llm_response.processing_time + formatting_time,
            'token_usage': llm_response.token_usage,
            'formatted_at': time.time()
        }
    
    def _determine_format_type(self, unified_input: UnifiedInput, content: str) -> str:
        """确定格式类型"""
        if unified_input.processor_type == 'comparison':
            return 'comparison_analysis'
        elif unified_input.processor_type == 'chitchat':
            return 'conversational'
        elif unified_input.has_multimodal_data():
            return 'multimodal_qa'
        elif len(content) > 500:
            return 'detailed_explanation'
        else:
            return 'simple_answer'
    
    def _create_error_formatted_response(self, llm_response: LLMResponse, unified_input: UnifiedInput, error: str) -> FormattedResponse:
        """创建错误格式化响应"""
        return FormattedResponse(
            content=llm_response.content,  # 返回原始内容
            metadata={
                'error': True,
                'formatting_error': error,
                'query': unified_input.query,
                'processor_type': unified_input.processor_type
            },
            format_type='error',
            processing_info={
                'llm_processing_time': llm_response.processing_time,
                'formatting_time': 0.0,
                'total_time': llm_response.processing_time,
                'token_usage': llm_response.token_usage
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_formatted = max(1, self.stats['total_formatted'])
        
        return {
            'total_formatted': self.stats['total_formatted'],
            'processor_distribution': self.stats['by_processor_type'].copy(),
            'performance': {
                'avg_formatting_time': self.stats['formatting_time'] / total_formatted,
                'total_formatting_time': self.stats['formatting_time']
            },
            'quality_metrics': {
                'truncated_responses': self.stats['truncated_responses'],
                'truncation_rate': self.stats['truncated_responses'] / total_formatted,
                'error_responses': self.stats['error_responses'],
                'error_rate': self.stats['error_responses'] / total_formatted
            },
            'configuration': self.settings
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_formatted': 0,
            'by_processor_type': {},
            'formatting_time': 0.0,
            'truncated_responses': 0,
            'error_responses': 0
        }

# =============================================================================
# 工厂函数
# =============================================================================

def create_response_formatter(config: Optional[Dict[str, Any]] = None) -> ResponseFormatter:
    """创建响应格式化器"""
    return ResponseFormatter(config)

# =============================================================================
# 测试和演示
# =============================================================================

if __name__ == "__main__":
    from .llm_engine import LLMResponse
    from .input_router import UnifiedInput
    
    print("📝 响应格式化器测试")
    
    # 创建格式化器
    formatter = create_response_formatter()
    
    # 创建测试数据
    test_llm_response = LLMResponse(
        content="科比·布莱恩特是洛杉矶湖人队的传奇球员。他在NBA生涯中获得了5次总冠军，被誉为篮球界的传奇人物。科比的职业精神和技术水平都达到了顶峰。",
        metadata={'model': 'test'},
        processing_time=1.5,
        token_usage={'input_tokens': 50, 'output_tokens': 100, 'total_tokens': 150}
    )
    
    test_unified_input = UnifiedInput(
        query="科比是谁？",
        processor_type="direct",
        text_context="科比相关信息"
    )
    
    # 测试格式化
    formatted_response = formatter.format_response(test_llm_response, test_unified_input)
    
    print(f"✅ 格式化完成")
    print(f"原始长度: {len(test_llm_response.content)}")
    print(f"格式化后长度: {len(formatted_response.content)}")
    print(f"格式类型: {formatted_response.format_type}")
    print(f"处理时间: {formatted_response.processing_info['formatting_time']:.3f}秒")
    
    # 显示统计信息
    print(f"\n📊 格式化器统计: {formatter.get_stats()}")
