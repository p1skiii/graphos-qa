"""
分词器
对英文查询进行分词、词性标注
"""
import spacy
from typing import List, Dict, Any
from dataclasses import dataclass
from app.core.schemas import QueryContext
from .base_processor import BaseNLPProcessor
import logging

logger = logging.getLogger(__name__)

@dataclass
class Token:
    """分词结果"""
    text: str           # 原文
    lemma: str          # 词根
    pos: str            # 词性
    tag: str            # 详细标签
    is_alpha: bool      # 是否为字母
    is_stop: bool       # 是否为停用词
    is_punct: bool      # 是否为标点
    ent_type: str       # 实体类型（如果有）

class Tokenizer(BaseNLPProcessor):
    """英文分词器 - 使用spaCy进行分词和词性标注"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__("tokenizer")
        self.model_name = model_name
        self.nlp = None
        
    def initialize(self) -> bool:
        """初始化spaCy模型"""
        try:
            self.nlp = spacy.load(self.model_name)
            self.initialized = True
            logger.info(f"✅ {self.name} 初始化成功 (模型: {self.model_name})")
            return True
        except OSError as e:
            logger.error(f"❌ {self.name} 初始化失败: 找不到模型 {self.model_name}")
            logger.error(f"请运行: python -m spacy download {self.model_name}")
            return False
        except Exception as e:
            logger.error(f"❌ {self.name} 初始化失败: {e}")
            return False
    
    def process(self, context: QueryContext) -> QueryContext:
        """
        分词并填充tokens字段，合并命名实体
        
        Args:
            context: 查询上下文
            
        Returns:
            QueryContext: 填充了tokens的上下文
        """
        self._add_trace(context, "start_tokenization")
        
        try:
            # 获取要处理的文本（优先使用翻译后的，否则使用原文）
            text = getattr(context, 'translated_query', None) or context.original_query
            
            # 使用spaCy处理
            doc = self.nlp(text)
            
            # 第一步：创建基础token列表
            base_tokens = []
            for token in doc:
                base_tokens.append(Token(
                    text=token.text,
                    lemma=token.lemma_,
                    pos=token.pos_,
                    tag=token.tag_,
                    is_alpha=token.is_alpha,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    ent_type=token.ent_type_ if token.ent_type_ else ""
                ))
            
            # 第二步：智能合并命名实体
            merged_tokens = self._merge_named_entities(doc, base_tokens)
            
            # 添加到context
            context.tokens = merged_tokens
            
            # 提取一些有用的统计信息
            content_tokens = [t for t in merged_tokens if t.is_alpha and not t.is_stop]
            question_words = [t for t in merged_tokens if t.lemma.lower() in ['who', 'what', 'when', 'where', 'why', 'how']]
            person_entities = [t for t in merged_tokens if t.ent_type == 'PERSON']
            
            # 添加追踪信息
            self._add_trace(context, "tokenization_complete", {
                "total_tokens": len(merged_tokens),
                "content_tokens": len(content_tokens),
                "question_words": [t.text for t in question_words],
                "person_entities": [t.text for t in person_entities],
                "text_processed": text[:100] + "..." if len(text) > 100 else text
            })
            
            logger.debug(f"🔤 分词完成: {len(merged_tokens)} tokens, {len(content_tokens)} content tokens, {len(person_entities)} person entities")
            
        except Exception as e:
            logger.error(f"❌ 分词失败: {e}")
            context.tokens = []
            self._add_trace(context, "tokenization_error", {"error": str(e)})
        
        return context
    
    def _merge_named_entities(self, doc, base_tokens: List[Token]) -> List[Token]:
        """
        智能合并spaCy识别的命名实体，特别是人名和组织名
        
        Args:
            doc: spaCy文档对象
            base_tokens: 基础token列表
            
        Returns:
            List[Token]: 合并后的token列表
        """
        if not doc.ents:
            return base_tokens
        
        merged_tokens = []
        processed_positions = set()
        
        # 处理所有实体，创建实体映射
        entity_map = {}
        for ent in doc.ents:
            # 只合并人名(PERSON)和组织(ORG)实体，过滤掉错误的识别
            if ent.label_ in ["PERSON", "ORG"]:
                # 清理实体文本，移除不合理的前缀
                clean_text = self._clean_entity_text(ent.text, ent.label_)
                if clean_text:
                    entity_map[ent.start] = {
                        'end': ent.end,
                        'text': clean_text,
                        'label': ent.label_
                    }
                    # 标记所有被合并的位置
                    for i in range(ent.start, ent.end):
                        processed_positions.add(i)
        
        # 重建token列表
        i = 0
        while i < len(base_tokens):
            if i in entity_map:
                # 创建合并的实体token
                entity_info = entity_map[i]
                merged_token = Token(
                    text=entity_info['text'],
                    lemma=entity_info['text'].lower(),
                    pos="PROPN",  # 专有名词
                    tag="NNP",    # 专有名词单数
                    is_alpha=True,
                    is_stop=False,
                    is_punct=False,
                    ent_type=entity_info['label']
                )
                merged_tokens.append(merged_token)
                # 跳到实体结束位置
                i = entity_info['end']
            elif i not in processed_positions:
                # 添加普通token
                merged_tokens.append(base_tokens[i])
                i += 1
            else:
                # 跳过被合并的token
                i += 1
        
        return merged_tokens
    
    def _clean_entity_text(self, text: str, label: str) -> str:
        """
        清理实体文本，移除不合理的部分
        
        Args:
            text: 原始实体文本
            label: 实体标签
            
        Returns:
            str: 清理后的文本，如果不合理则返回空字符串
        """
        text = text.strip()
        
        if label == "PERSON":
            # 对于人名，移除动词前缀
            unwanted_prefixes = ["compare", "tell", "show", "find"]
            text_lower = text.lower()
            
            for prefix in unwanted_prefixes:
                if text_lower.startswith(prefix + " "):
                    text = text[len(prefix):].strip()
                    break
            
            # 检查是否仍然是合理的人名（至少包含一个大写字母开头的单词）
            words = text.split()
            valid_words = [w for w in words if w and w[0].isupper() and w.isalpha()]
            if len(valid_words) >= 1:
                return " ".join(valid_words)
            else:
                return ""
        
        elif label == "ORG":
            # 对于组织名，简单清理
            if text and any(c.isupper() for c in text):
                return text
            else:
                return ""
        
        return text