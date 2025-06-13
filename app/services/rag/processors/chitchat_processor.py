"""
闲聊处理器 (Chitchat Processor)
用于处理 DOMAIN_CHITCHAT 类型的查询
适用于篮球领域的闲聊查询，如"聊聊篮球"、"NBA怎么样"、"你觉得哪个球员最厉害"
"""
from typing import Dict, Any, Optional, List
import random
import logging
from .base_processor import BaseProcessor, ProcessorUtils
from app.rag.components import ProcessorDefaultConfigs

logger = logging.getLogger(__name__)

class ChitchatProcessor(BaseProcessor):
    """闲聊处理器"""
    
    def __init__(self, config: Optional[ProcessorDefaultConfigs] = None):
        """初始化闲聊处理器"""
        if config is None:
            config = ProcessorDefaultConfigs.get_chitchat_processor_config()
        
        super().__init__(config)
        logger.info(f"💬 创建闲聊处理器")
        
        # 初始化闲聊资源
        self._init_chitchat_resources()
    
    def _init_chitchat_resources(self):
        """初始化闲聊资源"""
        # 话题模板
        self.chitchat_topics = {
            'general_basketball': [
                "篮球是一项很精彩的运动",
                "NBA有很多传奇球员",
                "篮球需要团队配合",
                "每个位置都有其独特作用"
            ],
            'player_discussion': [
                "每个球员都有自己的特点",
                "伟大的球员往往有独特的技能",
                "球员的成长历程都很励志",
                "不同时代的球员风格各异"
            ],
            'team_discussion': [
                "每支球队都有自己的文化",
                "球队化学反应很重要",
                "教练的作用不可忽视",
                "王朝球队都有其成功秘诀"
            ],
            'basketball_philosophy': [
                "篮球教会我们团队合作",
                "永不放弃是篮球精神",
                "篮球是艺术与竞技的结合",
                "每场比赛都有其独特魅力"
            ]
        }
        
        # 响应模板
        self.response_templates = {
            'opinion': [
                "我觉得{topic}很有意思。",
                "关于{topic}，我认为{opinion}。",
                "说到{topic}，{opinion}。"
            ],
            'question': [
                "你觉得{topic}怎么样？",
                "对于{topic}，你有什么看法？",
                "你最喜欢{topic}的哪个方面？"
            ],
            'suggestion': [
                "如果你对{topic}感兴趣，可以了解一下{suggestion}。",
                "关于{topic}，我建议{suggestion}。"
            ]
        }
        
        # 热门话题
        self.popular_topics = [
            "NBA历史最佳球员", "最经典的篮球比赛", "篮球技术发展",
            "球队管理哲学", "篮球文化影响", "青少年篮球发展"
        ]
    
    def _process_impl(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """处理闲聊查询"""
        logger.info(f"🔍 闲聊查询处理: {query}")
        
        try:
            # 1. 闲聊查询分析
            query_analysis = self._analyze_chitchat_query(query, context)
            
            # 2. 轻量级检索 (获取相关话题材料)
            retrieved_nodes = self.retriever.retrieve(
                query, 
                top_k=min(6, self.config.max_tokens // 200)
            )
            
            # 3. 构建简单上下文 (闲聊不需要复杂图结构)
            context_info = self._build_chitchat_context(retrieved_nodes, query_analysis)
            
            # 4. 生成闲聊回应
            chitchat_response = self._generate_chitchat_response(
                query, query_analysis, context_info
            )
            
            # 5. 添加互动元素
            interactive_response = self._add_interactive_elements(
                chitchat_response, query_analysis
            )
            
            # 6. 限制token数量
            final_text = ProcessorUtils.limit_tokens(
                interactive_response, 
                self.config.max_tokens
            )
            
            # 7. 构建结果
            result = {
                'success': True,
                'query': query,
                'context': context or {},
                'query_analysis': query_analysis,
                'retrieved_nodes_count': len(retrieved_nodes),
                'context_info': context_info,
                'contextualized_text': final_text,
                'processing_strategy': 'domain_chitchat',
                'confidence': self._calculate_confidence(query_analysis, context_info),
                'interaction_type': query_analysis.get('chitchat_type', 'general')
            }
            
            logger.info(f"✅ 闲聊查询处理完成，话题类型: {query_analysis.get('chitchat_type', 'general')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 闲聊查询处理失败: {str(e)}")
            raise
    
    def _analyze_chitchat_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析闲聊查询"""
        analysis = {
            'query_type': 'domain_chitchat',
            'chitchat_type': self._detect_chitchat_type(query),
            'emotional_tone': self._detect_emotional_tone(query),
            'topic_focus': self._extract_topic_focus(query),
            'interaction_intent': self._detect_interaction_intent(query),
            'formality_level': self._detect_formality_level(query)
        }
        
        # 分析上下文
        if context:
            analysis['conversation_history'] = context.get('conversation_history', [])
            analysis['user_preferences'] = context.get('preferences', {})
        
        return analysis
    
    def _detect_chitchat_type(self, query: str) -> str:
        """检测闲聊类型"""
        query_lower = query.lower()
        
        # 观点分享
        if any(word in query for word in ['觉得', '认为', '看法', '观点']):
            return 'opinion_sharing'
        
        # 话题探讨
        elif any(word in query for word in ['聊聊', '谈谈', '说说', '讨论']):
            return 'topic_discussion'
        
        # 推荐询问
        elif any(word in query for word in ['推荐', '建议', '介绍']):
            return 'recommendation'
        
        # 比较讨论
        elif any(word in query for word in ['比较', '对比', '更喜欢']):
            return 'comparison_discussion'
        
        # 知识分享
        elif any(word in query for word in ['知道', '了解', '听说']):
            return 'knowledge_sharing'
        
        # 情感表达
        elif any(word in query for word in ['喜欢', '讨厌', '最爱', '最恨']):
            return 'emotional_expression'
        
        return 'general'
    
    def _detect_emotional_tone(self, query: str) -> str:
        """检测情感色调"""
        query_lower = query.lower()
        
        # 积极情感
        positive_words = ['喜欢', '最爱', '精彩', '厉害', '棒', '好']
        if any(word in query for word in positive_words):
            return 'positive'
        
        # 消极情感
        negative_words = ['讨厌', '差', '糟糕', '无聊']
        if any(word in query for word in negative_words):
            return 'negative'
        
        # 好奇/疑问
        curious_words = ['为什么', '怎么', '如何', '吗', '呢']
        if any(word in query for word in curious_words):
            return 'curious'
        
        return 'neutral'
    
    def _extract_topic_focus(self, query: str) -> List[str]:
        """提取话题焦点"""
        topics = []
        query_lower = query.lower()
        
        # 球员相关
        if any(word in query for word in ['球员', '科比', '詹姆斯', '乔丹']):
            topics.append('player')
        
        # 球队相关
        if any(word in query for word in ['球队', '湖人', '勇士', '公牛']):
            topics.append('team')
        
        # 比赛相关
        if any(word in query for word in ['比赛', '季后赛', '总决赛']):
            topics.append('game')
        
        # 技术相关
        if any(word in query for word in ['技术', '战术', '投篮', '防守']):
            topics.append('technique')
        
        # 历史相关
        if any(word in query for word in ['历史', '传奇', '经典', '过去']):
            topics.append('history')
        
        # NBA相关
        if 'nba' in query_lower or 'NBA' in query:
            topics.append('nba')
        
        return topics if topics else ['general']
    
    def _detect_interaction_intent(self, query: str) -> str:
        """检测互动意图"""
        query_lower = query.lower()
        
        # 寻求建议
        if any(word in query for word in ['建议', '推荐', '该', '应该']):
            return 'seeking_advice'
        
        # 寻求观点
        elif any(word in query for word in ['觉得', '认为', '看法', '想法']):
            return 'seeking_opinion'
        
        # 分享经历
        elif any(word in query for word in ['我', '我的', '曾经', '以前']):
            return 'sharing_experience'
        
        # 提问探讨
        elif any(word in query for word in ['为什么', '怎么', '如何']):
            return 'asking_question'
        
        return 'general_chat'
    
    def _detect_formality_level(self, query: str) -> str:
        """检测正式程度"""
        # 简单的正式程度检测
        informal_indicators = ['呀', '哈', '嘿', '哇', '额']
        if any(indicator in query for indicator in informal_indicators):
            return 'informal'
        
        formal_indicators = ['请问', '请', '您', '敬请']
        if any(indicator in query for indicator in formal_indicators):
            return 'formal'
        
        return 'neutral'
    
    def _build_chitchat_context(self, retrieved_nodes: List[Dict], 
                               query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """构建闲聊上下文"""
        context_info = {
            'relevant_entities': [],
            'topic_materials': [],
            'conversation_starters': [],
            'knowledge_points': []
        }
        
        # 提取相关实体
        if retrieved_nodes:
            for node in retrieved_nodes[:4]:  # 最多4个实体
                entity_info = {
                    'name': node.get('name', '未知'),
                    'type': node.get('type', '未知'),
                    'description': node.get('description', ''),
                    'similarity': node.get('similarity', 0.0)
                }
                context_info['relevant_entities'].append(entity_info)
        
        # 根据话题焦点准备话题材料
        topic_focus = query_analysis.get('topic_focus', ['general'])
        for topic in topic_focus:
            if topic in self.chitchat_topics:
                materials = self.chitchat_topics[topic]
                context_info['topic_materials'].extend(random.sample(materials, min(2, len(materials))))
        
        # 生成对话启发点
        context_info['conversation_starters'] = self._generate_conversation_starters(query_analysis)
        
        # 准备知识点
        context_info['knowledge_points'] = self._prepare_knowledge_points(retrieved_nodes)
        
        return context_info
    
    def _generate_conversation_starters(self, query_analysis: Dict[str, Any]) -> List[str]:
        """生成对话启发点"""
        starters = []
        chitchat_type = query_analysis.get('chitchat_type', 'general')
        topic_focus = query_analysis.get('topic_focus', ['general'])
        
        if chitchat_type == 'opinion_sharing':
            starters.extend([
                "每个人对篮球都有不同的理解",
                "观点分享让我们看到更多角度",
                "你的看法很有意思"
            ])
        
        elif chitchat_type == 'topic_discussion':
            starters.extend([
                "这个话题确实值得深入探讨",
                "让我们从不同角度来看看",
                "这是个很有趣的讨论点"
            ])
        
        if 'player' in topic_focus:
            starters.append("每个球员都有独特的故事")
        
        if 'team' in topic_focus:
            starters.append("球队文化往往决定风格")
        
        return starters[:3]  # 返回最多3个启发点
    
    def _prepare_knowledge_points(self, retrieved_nodes: List[Dict]) -> List[str]:
        """准备知识点"""
        knowledge_points = []
        
        if retrieved_nodes:
            for node in retrieved_nodes[:3]:
                name = node.get('name', '')
                node_type = node.get('type', '')
                
                if node_type == 'player':
                    knowledge_points.append(f"{name}是一位知名球员")
                elif node_type == 'team':
                    knowledge_points.append(f"{name}是一支重要球队")
        
        return knowledge_points
    
    def _generate_chitchat_response(self, query: str, query_analysis: Dict[str, Any], 
                                  context_info: Dict[str, Any]) -> str:
        """生成闲聊回应"""
        response_parts = []
        
        chitchat_type = query_analysis.get('chitchat_type', 'general')
        emotional_tone = query_analysis.get('emotional_tone', 'neutral')
        topic_focus = query_analysis.get('topic_focus', ['general'])
        
        # 根据闲聊类型生成开场
        opening = self._generate_opening(chitchat_type, emotional_tone)
        if opening:
            response_parts.append(opening)
        
        # 添加相关内容
        content = self._generate_content(query_analysis, context_info)
        if content:
            response_parts.append(content)
        
        # 添加互动元素
        interaction = self._generate_interaction_element(query_analysis)
        if interaction:
            response_parts.append(interaction)
        
        return "\n\n".join(response_parts)
    
    def _generate_opening(self, chitchat_type: str, emotional_tone: str) -> str:
        """生成开场白"""
        openings = {
            'opinion_sharing': {
                'positive': "很高兴你愿意分享观点！",
                'negative': "理解你的想法，每个人都有不同看法。",
                'neutral': "关于这个话题，确实有很多角度可以探讨。",
                'curious': "这是个很好的观察角度！"
            },
            'topic_discussion': {
                'positive': "这个话题很有意思！",
                'negative': "即使是争议性话题也值得讨论。",
                'neutral': "让我们来聊聊这个话题。",
                'curious': "你提到了一个很好的讨论点。"
            },
            'recommendation': {
                'positive': "很乐意给你一些推荐！",
                'negative': "虽然选择有限，但还是有一些建议。",
                'neutral': "我来为你推荐一些相关内容。",
                'curious': "让我想想有什么好的推荐。"
            }
        }
        
        type_openings = openings.get(chitchat_type, {})
        return type_openings.get(emotional_tone, "让我们聊聊篮球吧！")
    
    def _generate_content(self, query_analysis: Dict[str, Any], 
                         context_info: Dict[str, Any]) -> str:
        """生成主要内容"""
        content_parts = []
        
        # 使用检索到的实体信息
        relevant_entities = context_info.get('relevant_entities', [])
        if relevant_entities:
            entity_mentions = []
            for entity in relevant_entities[:3]:
                name = entity.get('name', '')
                entity_type = entity.get('type', '')
                if name and entity_type:
                    entity_mentions.append(f"{name}({entity_type})")
            
            if entity_mentions:
                content_parts.append(f"说到这个话题，{', '.join(entity_mentions)} 都是很好的例子。")
        
        # 添加话题材料
        topic_materials = context_info.get('topic_materials', [])
        if topic_materials:
            selected_material = random.choice(topic_materials)
            content_parts.append(selected_material)
        
        # 添加知识点
        knowledge_points = context_info.get('knowledge_points', [])
        if knowledge_points:
            content_parts.append(random.choice(knowledge_points))
        
        return " ".join(content_parts) if content_parts else "篮球世界总是充满惊喜和讨论点。"
    
    def _generate_interaction_element(self, query_analysis: Dict[str, Any]) -> str:
        """生成互动元素"""
        interaction_intent = query_analysis.get('interaction_intent', 'general_chat')
        topic_focus = query_analysis.get('topic_focus', ['general'])
        
        interactions = []
        
        if interaction_intent == 'seeking_opinion':
            interactions.extend([
                "你觉得呢？",
                "你的看法是什么？",
                "对此你有什么想法？"
            ])
        
        elif interaction_intent == 'seeking_advice':
            interactions.extend([
                "希望这些建议对你有帮助。",
                "你可以根据自己的兴趣选择。",
                "还有什么其他想了解的吗？"
            ])
        
        if 'player' in topic_focus:
            interactions.append("你有最喜欢的球员吗？")
        
        if 'team' in topic_focus:
            interactions.append("你支持哪支球队？")
        
        if interactions:
            return random.choice(interactions)
        
        return "你还想聊什么篮球话题？"
    
    def _add_interactive_elements(self, response: str, query_analysis: Dict[str, Any]) -> str:
        """添加互动元素"""
        # 根据正式程度调整语气
        formality_level = query_analysis.get('formality_level', 'neutral')
        
        if formality_level == 'informal':
            # 添加更轻松的表达
            response = response.replace("。", "~")
            if not response.endswith(('？', '！', '~')):
                response += " 😊"
        
        elif formality_level == 'formal':
            # 保持正式语气
            if not response.endswith(('。', '？', '！')):
                response += "。"
        
        # 添加话题延续
        topic_focus = query_analysis.get('topic_focus', ['general'])
        if len(topic_focus) > 1:
            response += f"\n\n顺便问一下，在{topic_focus[0]}和{topic_focus[1]}中，你更感兴趣哪个方面？"
        
        return response
    
    def _calculate_confidence(self, query_analysis: Dict[str, Any], 
                            context_info: Dict[str, Any]) -> float:
        """计算闲聊置信度"""
        # 闲聊的置信度主要基于话题匹配度
        base_confidence = 0.8  # 闲聊基础置信度较高
        
        # 话题匹配度加成
        topic_focus = query_analysis.get('topic_focus', [])
        if 'general' not in topic_focus:  # 有具体话题焦点
            topic_bonus = 0.1
        else:
            topic_bonus = 0
        
        # 上下文信息加成
        relevant_entities = context_info.get('relevant_entities', [])
        context_bonus = min(0.1, len(relevant_entities) * 0.025)
        
        total_confidence = base_confidence + topic_bonus + context_bonus
        return min(1.0, total_confidence)

# =============================================================================
# 工厂函数
# =============================================================================

def create_chitchat_processor(custom_config: Optional[Dict[str, Any]] = None) -> ChitchatProcessor:
    """创建闲聊处理器实例"""
    if custom_config:
        config = ProcessorDefaultConfigs.get_chitchat_processor_config()
        
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
        
        return ChitchatProcessor(config)
    else:
        return ChitchatProcessor()
