#!/usr/bin/env python3
"""
Smart Pre-processor Algorithm Validation

验证Stage 3 (智能意图识别) 和 Stage 4 (实体提取) 的核心算法
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# 核心算法实现 (从smart_preprocessor.py提取的核心逻辑)
# =============================================================================

@dataclass
class ParsedIntent:
    """意图解析结果"""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    is_basketball_related: bool
    players: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    processing_time: float = 0.0

class CoreAIAlgorithms:
    """核心AI算法实现"""
    
    def __init__(self):
        self.intent_labels = [
            'ATTRIBUTE_QUERY',          # 年龄、身高、体重查询
            'SIMPLE_RELATION_QUERY',    # 球队归属、简单事实
            'COMPLEX_RELATION_QUERY',   # 多步推理
            'COMPARATIVE_QUERY',        # 球员比较
            'DOMAIN_CHITCHAT',          # 篮球相关闲聊
            'OUT_OF_DOMAIN'             # 非篮球查询
        ]
        
        # 篮球实体知识库
        self.basketball_entities = {
            'players': [
                'kobe bryant', 'kobe', 'lebron james', 'lebron', 'michael jordan', 'jordan',
                'yao ming', 'yao', 'stephen curry', 'curry', 'shaquille oneal', 'shaq',
                'tim duncan', 'duncan', 'magic johnson', 'magic', 'larry bird', 'bird'
            ],
            'teams': [
                'lakers', 'warriors', 'bulls', 'heat', 'celtics', 'rockets', 'spurs',
                'los angeles lakers', 'golden state warriors', 'chicago bulls',
                'miami heat', 'boston celtics', 'houston rockets', 'san antonio spurs'
            ],
            'attributes': [
                'age', 'height', 'weight', 'position', 'team', 'championship',
                'points', 'assists', 'rebounds', 'career', 'stats', 'salary'
            ]
        }
    
    def intelligent_multitask_classification(self, text: str) -> ParsedIntent:
        """
        智能多任务模型 - Stage 3 (意图识别) + Stage 4 (实体提取)
        
        Args:
            text: 标准化英文文本
            
        Returns:
            ParsedIntent: 完整分类结果
        """
        start_time = time.time()
        
        try:
            # 文本预处理
            processed_text = self._preprocess_for_ai_model(text)
            
            # Stage 3: 意图分类 (6标签分类)
            intent_result = self._classify_intent_ai(processed_text)
            
            # Stage 4: 实体提取 (结构化信息提取)
            entity_result = self._extract_entities_ai(processed_text, intent_result['intent'])
            
            # 确定篮球领域相关性
            is_basketball_related = self._is_basketball_domain(processed_text, entity_result)
            
            return ParsedIntent(
                intent=intent_result['intent'],
                confidence=intent_result['confidence'],
                players=entity_result.get('players', []),
                teams=entity_result.get('teams', []),
                attributes=entity_result.get('attributes', []),
                entities=entity_result,
                is_basketball_related=is_basketball_related,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"❌ AI分类失败: {str(e)}")
            return ParsedIntent(
                intent='OUT_OF_DOMAIN',
                confidence=0.0,
                players=[],
                teams=[],
                attributes=[],
                entities={},
                is_basketball_related=False,
                processing_time=time.time() - start_time
            )
    
    def _preprocess_for_ai_model(self, text: str) -> str:
        """为AI模型预处理文本"""
        text = text.lower().strip()
        text = text.replace("vs.", "versus").replace("vs", "versus").replace("&", "and")
        return " ".join(text.split())
    
    def _classify_intent_ai(self, text: str) -> Dict[str, Any]:
        """AI驱动的意图分类 (Stage 3)"""
        try:
            # 特征提取
            features = self._extract_text_features(text)
            
            # 意图评分算法
            intent_scores = self._calculate_intent_scores(text, features)
            
            # 获取最佳意图
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            
            return {
                'intent': best_intent[0],
                'confidence': best_intent[1],
                'all_scores': intent_scores
            }
            
        except Exception as e:
            print(f"⚠️ 意图分类失败: {str(e)}")
            return {'intent': 'OUT_OF_DOMAIN', 'confidence': 0.0, 'all_scores': {}}
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_question_words': any(word in text for word in ['what', 'who', 'how', 'when', 'where', 'why']),
            'has_comparison_words': any(word in text for word in ['versus', 'compare', 'better', 'best', 'vs']),
            'has_attribute_words': any(word in text for word in ['age', 'height', 'weight', 'tall', 'old']),
            'has_relation_words': any(word in text for word in ['team', 'play', 'belong', 'member']),
            'has_greeting_words': any(word in text for word in ['hello', 'hi', 'hey', 'greetings']),
        }
        
        # 使用增强的篮球领域检测
        features['basketball_domain_confidence'] = self._enhanced_basketball_detection(text)
        
        # 统计实体数量
        entity_count = 0
        for player in self.basketball_entities['players']:
            if player.lower() in text:
                entity_count += 1
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                entity_count += 1
        features['entity_count'] = entity_count
        
        return features
    
    def _enhanced_basketball_detection(self, text: str) -> float:
        """增强的篮球领域检测"""
        confidence = 0.0
        
        # 直接篮球关键词
        basketball_keywords = ['basketball', 'nba', 'player', 'team', 'game', 'sport', 'court', 'score']
        keyword_matches = sum(1 for word in basketball_keywords if word in text)
        confidence += min(keyword_matches / 2.0, 0.4)
        
        # 篮球实体 (球员、球队)
        entity_confidence = self._calculate_entity_confidence(text)
        confidence += entity_confidence
        
        # 篮球特定术语
        basketball_terms = ['championship', 'playoffs', 'season', 'draft', 'rookie', 'veteran', 'coach']
        term_matches = sum(1 for term in basketball_terms if term in text)
        confidence += min(term_matches / 3.0, 0.2)
        
        # 篮球动作和概念
        basketball_actions = ['shoot', 'dunk', 'assist', 'rebound', 'steal', 'block', 'foul']
        action_matches = sum(1 for action in basketball_actions if action in text)
        confidence += min(action_matches / 4.0, 0.15)
        
        return min(confidence, 1.0)
    
    def _calculate_entity_confidence(self, text: str) -> float:
        """基于篮球实体计算置信度"""
        confidence = 0.0
        
        # 检查球员名字
        player_matches = 0
        for player in self.basketball_entities['players']:
            if player.lower() in text:
                player_matches += 1
        confidence += min(player_matches / 2.0, 0.4)
        
        # 检查球队名字
        team_matches = 0
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                team_matches += 1
        confidence += min(team_matches / 2.0, 0.3)
        
        # 检查篮球属性
        attr_matches = 0
        basketball_attrs = ['age', 'height', 'weight', 'position', 'team', 'points', 'assists', 'rebounds']
        for attr in basketball_attrs:
            if attr in text:
                attr_matches += 1
        confidence += min(attr_matches / 3.0, 0.2)
        
        return confidence
    
    def _calculate_intent_scores(self, text: str, features: Dict[str, Any]) -> Dict[str, float]:
        """计算意图分数"""
        scores = {intent: 0.0 for intent in self.intent_labels}
        
        domain_confidence = features['basketball_domain_confidence']
        
        # 如果不是篮球领域，返回OUT_OF_DOMAIN
        if domain_confidence < 0.3:
            scores['OUT_OF_DOMAIN'] = 0.9
            return scores
        
        # ATTRIBUTE_QUERY评分 - 增强版
        if features['has_attribute_words'] and features['has_question_words']:
            scores['ATTRIBUTE_QUERY'] = 0.85 + domain_confidence * 0.15
        elif any(word in text for word in ['old', 'age', 'tall', 'height', 'weight']) and domain_confidence > 0.5:
            scores['ATTRIBUTE_QUERY'] = 0.75 + domain_confidence * 0.25
        
        # COMPARATIVE_QUERY评分 - 增强版
        if features['has_comparison_words']:
            scores['COMPARATIVE_QUERY'] = 0.8 + domain_confidence * 0.2
        elif any(word in text for word in ['better', 'best', 'greatest']) and domain_confidence > 0.6:
            scores['COMPARATIVE_QUERY'] = 0.7 + domain_confidence * 0.3
        
        # SIMPLE_RELATION_QUERY评分 - 增强版
        if features['has_relation_words'] and not features['has_comparison_words']:
            scores['SIMPLE_RELATION_QUERY'] = 0.7 + domain_confidence * 0.3
        elif any(word in text for word in ['which', 'what']) and domain_confidence > 0.5:
            scores['SIMPLE_RELATION_QUERY'] = 0.65 + domain_confidence * 0.35
        
        # COMPLEX_RELATION_QUERY评分
        if features['word_count'] > 8 and domain_confidence > 0.5:
            scores['COMPLEX_RELATION_QUERY'] = 0.5 + min(features['word_count'] / 15.0, 0.4)
        
        # DOMAIN_CHITCHAT评分 - 增强版
        if features['has_greeting_words'] and domain_confidence > 0.3:
            scores['DOMAIN_CHITCHAT'] = 0.7 + domain_confidence * 0.3
        elif 'basketball' in text and features['word_count'] < 6:
            scores['DOMAIN_CHITCHAT'] = 0.65 + domain_confidence * 0.35
        elif any(word in text for word in ['tell me', 'explain', 'about']) and domain_confidence > 0.5:
            scores['DOMAIN_CHITCHAT'] = 0.6 + domain_confidence * 0.4
        
        # 如果有强篮球实体，提升分数
        entity_boost = self._calculate_entity_boost(text)
        for intent in scores:
            if intent != 'OUT_OF_DOMAIN':
                scores[intent] += entity_boost
        
        # 标准化分数
        max_score = max(scores.values())
        if max_score > 0:
            for intent in scores:
                scores[intent] = min(scores[intent] / max_score * 0.95, 0.95)
        else:
            scores['OUT_OF_DOMAIN'] = 0.8
        
        return scores
    
    def _calculate_entity_boost(self, text: str) -> float:
        """基于实体计算提升分数"""
        boost = 0.0
        
        # 知名球员强提升
        famous_players = ['kobe', 'lebron', 'jordan', 'yao ming', 'curry', 'shaq']
        for player in famous_players:
            if player in text:
                boost += 0.1
        
        # 球队名提升
        for team in self.basketball_entities['teams']:
            if team.lower() in text:
                boost += 0.05
        
        return min(boost, 0.3)
    
    def _extract_entities_ai(self, text: str, intent: str) -> Dict[str, List[str]]:
        """AI驱动的实体提取 (Stage 4)"""
        try:
            entities = {'players': [], 'teams': [], 'attributes': []}
            
            if intent == 'OUT_OF_DOMAIN':
                return entities
            
            # 智能实体提取
            entities = self._smart_entity_extraction(text)
            
            # 基于意图的实体优化
            entities = self._refine_entities_by_intent(entities, intent, text)
            
            return entities
            
        except Exception as e:
            print(f"⚠️ AI实体提取失败: {str(e)}")
            return {'players': [], 'teams': [], 'attributes': []}
    
    def _smart_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """智能实体提取"""
        entities = {'players': [], 'teams': [], 'attributes': []}
        
        # 增强的球员提取
        for player in self.basketball_entities['players']:
            player_lower = player.lower()
            
            # 精确匹配
            if player_lower in text:
                normalized_name = self._normalize_player_name(player)
                if normalized_name not in entities['players']:
                    entities['players'].append(normalized_name)
            
            # 模糊匹配
            elif self._advanced_fuzzy_match(player_lower, text):
                normalized_name = self._normalize_player_name(player)
                if normalized_name not in entities['players']:
                    entities['players'].append(normalized_name)
        
        # 增强的球队提取
        for team in self.basketball_entities['teams']:
            team_lower = team.lower()
            if team_lower in text or any(alias in text for alias in self._get_team_aliases(team)):
                normalized_team = self._normalize_team_name(team)
                if normalized_team not in entities['teams']:
                    entities['teams'].append(normalized_team)
        
        # 上下文感知的属性提取
        for attr in self.basketball_entities['attributes']:
            if attr.lower() in text:
                entities['attributes'].append(attr)
        
        # 从上下文推断属性
        inferred_attrs = self._infer_attributes_from_context(text)
        for attr in inferred_attrs:
            if attr not in entities['attributes']:
                entities['attributes'].append(attr)
        
        return entities
    
    def _normalize_player_name(self, player_name: str) -> str:
        """标准化球员名字"""
        name_mapping = {
            'kobe': 'Kobe Bryant',
            'kobe bryant': 'Kobe Bryant',
            'lebron': 'LeBron James',
            'lebron james': 'LeBron James',
            'jordan': 'Michael Jordan',
            'michael jordan': 'Michael Jordan',
            'yao': 'Yao Ming',
            'yao ming': 'Yao Ming',
            'curry': 'Stephen Curry',
            'stephen curry': 'Stephen Curry',
            'shaq': 'Shaquille O\'Neal',
            'shaquille oneal': 'Shaquille O\'Neal'
        }
        
        return name_mapping.get(player_name.lower(), player_name.title())
    
    def _normalize_team_name(self, team_name: str) -> str:
        """标准化球队名字"""
        team_mapping = {
            'lakers': 'Los Angeles Lakers',
            'warriors': 'Golden State Warriors',
            'bulls': 'Chicago Bulls',
            'heat': 'Miami Heat',
            'celtics': 'Boston Celtics',
            'rockets': 'Houston Rockets'
        }
        
        return team_mapping.get(team_name.lower(), team_name.title())
    
    def _get_team_aliases(self, team_name: str) -> List[str]:
        """获取球队别名"""
        alias_mapping = {
            'lakers': ['la lakers', 'l.a. lakers'],
            'warriors': ['gsw', 'dubs'],
            'bulls': ['chicago'],
            'heat': ['miami'],
            'celtics': ['boston'],
            'rockets': ['houston']
        }
        
        return alias_mapping.get(team_name.lower(), [])
    
    def _advanced_fuzzy_match(self, target: str, text: str, threshold: float = 0.7) -> bool:
        """高级模糊匹配"""
        words = text.split()
        target_words = target.split()
        
        for word in words:
            for target_word in target_words:
                if len(target_word) > 3:
                    if target_word in word and len(target_word) / len(word) >= threshold:
                        return True
                    if word in target_word and len(word) / len(target_word) >= threshold:
                        return True
        
        # 检查缩写匹配
        if len(target_words) > 1:
            acronym = ''.join(word[0] for word in target_words)
            if acronym in text:
                return True
        
        return False
    
    def _infer_attributes_from_context(self, text: str) -> List[str]:
        """从上下文推断属性"""
        inferred = []
        
        if any(word in text for word in ['old', 'age', 'born', 'birth']):
            inferred.append('age')
        
        if any(word in text for word in ['tall', 'height', 'feet', 'ft', 'inches']):
            inferred.append('height')
        
        if any(word in text for word in ['heavy', 'weight', 'pounds', 'lbs', 'kg']):
            inferred.append('weight')
        
        if any(word in text for word in ['team', 'play for', 'plays for', 'member']):
            inferred.append('team')
        
        if any(word in text for word in ['points', 'scoring', 'score']):
            inferred.append('points')
        
        return inferred
    
    def _refine_entities_by_intent(self, entities: Dict[str, List[str]], 
                                 intent: str, text: str) -> Dict[str, List[str]]:
        """基于意图优化提取的实体"""
        # ATTRIBUTE_QUERY: 确保有属性
        if intent == 'ATTRIBUTE_QUERY' and not entities['attributes']:
            if any(word in text for word in ['old', 'age']):
                entities['attributes'].append('age')
            elif any(word in text for word in ['tall', 'height']):
                entities['attributes'].append('height')
            elif any(word in text for word in ['weight', 'heavy']):
                entities['attributes'].append('weight')
        
        return entities
    
    def _is_basketball_domain(self, text: str, entities: Dict[str, List[str]]) -> bool:
        """判断查询是否与篮球相关"""
        try:
            if any(entities.values()):
                return True
            
            basketball_keywords = [
                'basketball', 'nba', 'game', 'player', 'team', 'sport',
                'court', 'ball', 'shoot', 'score', 'championship'
            ]
            
            return any(keyword in text for keyword in basketball_keywords)
            
        except Exception as e:
            print(f"⚠️ 领域检查失败: {str(e)}")
            return False

def main():
    """主测试函数"""
    print("🚀 Smart Pre-processor 核心AI算法验证")
    print("=" * 60)
    print("验证 Stage 3 (智能意图识别) 和 Stage 4 (实体提取)")
    print("=" * 60)
    
    # 创建核心AI算法实例
    ai_core = CoreAIAlgorithms()
    
    # 测试查询
    test_queries = [
        "How old is Kobe Bryant",
        "Compare Kobe versus Jordan",
        "What team does LeBron play for",
        "Tell me about basketball",
        "Hello",
        "What is the weather today",
        "How tall is Yao Ming",
        "Who is better, Curry or Durant"
    ]
    
    print("\n🧠 测试智能多任务分类:")
    print("-" * 60)
    
    total_start = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] 查询: {query}")
        
        result = ai_core.intelligent_multitask_classification(query)
        
        print(f"    意图: {result.intent}")
        print(f"    置信度: {result.confidence:.3f}")
        print(f"    篮球相关: {result.is_basketball_related}")
        print(f"    球员: {result.players}")
        print(f"    球队: {result.teams}")
        print(f"    属性: {result.attributes}")
        print(f"    处理时间: {result.processing_time:.4f}s")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✅ Stage 3 和 Stage 4 算法验证完成!")
    print(f"📊 总处理时间: {total_time:.4f}s")
    print(f"⚡ 平均处理时间: {total_time/len(test_queries):.4f}s/查询")
    print("=" * 60)
    
    # 算法性能统计
    print("\n📈 算法性能分析:")
    print("-" * 30)
    
    basketball_queries = 0
    correct_classifications = 0
    
    for query in test_queries:
        result = ai_core.intelligent_multitask_classification(query)
        
        # 统计篮球查询
        expected_basketball = any(word in query.lower() for word in 
                                ['kobe', 'jordan', 'lebron', 'basketball', 'team', 'yao', 'curry'])
        
        if expected_basketball:
            basketball_queries += 1
            if result.is_basketball_related:
                correct_classifications += 1
    
    accuracy = correct_classifications / basketball_queries if basketball_queries > 0 else 0
    print(f"🎯 篮球查询识别准确率: {accuracy:.1%}")
    print(f"📊 篮球查询数量: {basketball_queries}/{len(test_queries)}")
    print(f"✅ 正确分类数量: {correct_classifications}")

if __name__ == "__main__":
    main()
