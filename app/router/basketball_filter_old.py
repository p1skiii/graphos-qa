"""
篮球领域过滤器 - IF-ELSE规则判断是否属于篮球领域
"""
import re
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class BasketballDomainFilter:
    """篮球领域过滤器"""
    
    def __init__(self):
        # 篮球相关关键词（中英文）
        self.basketball_keywords = [
            # 中文关键词
            '篮球', 'NBA', '球员', '球队', '教练', '比赛', '得分', '篮板', '助攻',
            '总冠军', '季后赛', '常规赛', '全明星', '选秀', '新秀', '退役',
            '姚明', '科比', '詹姆斯', '乔丹', '邓肯', '奥尼尔', '库里', '杜兰特',
            '湖人', '勇士', '火箭', '马刺', '凯尔特人', '公牛', '热火', '骑士',
            '身高', '体重', '年龄', '位置', '中锋', '前锋', '后卫', '队友',
            '多少岁', '多高', '多重', '哪个队', '什么关系', '谁更强',
            
            # 英文关键词
            'basketball', 'nba', 'player', 'team', 'coach', 'game', 'score', 
            'point', 'rebound', 'assist', 'steal', 'block', 'championship', 
            'playoff', 'season', 'draft', 'rookie', 'retire', 'height', 
            'weight', 'age', 'position', 'center', 'guard', 'forward',
            'teammate', 'relationship',
            
            # 知名球员（英文）
            'yao ming', 'kobe', 'lebron', 'jordan', 'duncan', 'shaq',
            'curry', 'durant', 'wade', 'paul', 'harden', 'westbrook',
            
            # 知名球队（英文）
            'lakers', 'warriors', 'rockets', 'spurs', 'celtics', 'bulls',
            'heat', 'cavaliers', 'knicks', 'clippers', 'nuggets',
            
            # 城市（与球队相关）
            'los angeles', 'golden state', 'houston', 'san antonio',
            'boston', 'chicago', 'miami', 'cleveland', 'new york',
            
            # 常见问法
            'how old', 'how tall', 'how heavy', 'which team', 'what position',
            'what relationship', 'how many', 'when born', 'where from'
        ]
        
        # 明确的非篮球关键词
        self.non_basketball_keywords = [
            # 其他运动
            '足球', '网球', '高尔夫', '游泳', '棒球', '梅西', '罗纳尔多', '世界杯',
            'football', 'soccer', 'tennis', 'golf', 'baseball', 'swimming',
            'messi', 'ronaldo', 'federer', 'world cup', 'fifa',
            
            # 日常话题
            '天气', '下雨', '温度', '你好', '再见', '名字', '怎么样',
            'weather', 'rain', 'sunny', 'temperature', 'hello', 'goodbye',
            'what is your name', 'how are you', 'good morning',
            
            # 技术话题
            '电脑', '编程', '软件', '网站', '股市', '房价', '餐厅', '美食',
            'computer', 'python', 'programming', 'software', 'website',
            'stock market', 'money', 'price', 'restaurant', 'food',
            
            # 娱乐话题
            '电影', '音乐', '游戏', '明星', '电视剧',
            'movie', 'music', 'game', 'celebrity', 'tv show'
        ]
        
        # 统计信息
        self.stats = {
            'total_filtered': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0
        }
    
    def is_basketball_domain(self, text: str) -> Tuple[bool, str, Dict]:
        """判断是否属于篮球领域"""
        self.stats['total_filtered'] += 1
        
        text_lower = text.lower().strip()
        
        # 基础检查
        if len(text_lower) < 2:
            self.stats['non_basketball_filtered'] += 1
            return False, "text too short", {'length': len(text_lower)}
        
        # 检查篮球关键词匹配
        basketball_matches = []
        for keyword in self.basketball_keywords:
            if keyword in text_lower:
                basketball_matches.append(keyword)
        
        # 检查非篮球关键词匹配
        non_basketball_matches = []
        for keyword in self.non_basketball_keywords:
            if keyword in text_lower:
                non_basketball_matches.append(keyword)
        
        # 详细分析结果
        analysis = {
            'basketball_matches': basketball_matches,
            'non_basketball_matches': non_basketball_matches,
            'basketball_score': len(basketball_matches),
            'non_basketball_score': len(non_basketball_matches),
            'text_length': len(text_lower)
        }
        
        # 决策逻辑
        if basketball_matches and not non_basketball_matches:
            # 有篮球关键词，无非篮球关键词 -> 篮球领域
            self.stats['basketball_passed'] += 1
            return True, f"basketball keywords: {basketball_matches[:3]}", analysis
        
        elif non_basketball_matches and not basketball_matches:
            # 有非篮球关键词，无篮球关键词 -> 非篮球领域
            self.stats['non_basketball_filtered'] += 1
            return False, f"non-basketball keywords: {non_basketball_matches[:3]}", analysis
        
        elif basketball_matches and non_basketball_matches:
            # 两者都有，比较数量
            if len(basketball_matches) > len(non_basketball_matches):
                self.stats['basketball_passed'] += 1
                return True, f"basketball dominant: {basketball_matches[:3]}", analysis
            else:
                self.stats['non_basketball_filtered'] += 1
                return False, f"non-basketball dominant: {non_basketball_matches[:3]}", analysis
        
        else:
            # 都没有明确关键词，使用启发式规则
            
            # 检查是否包含人名模式（可能是球员名字）
            if self._has_person_name_pattern(text_lower):
                self.stats['basketball_passed'] += 1
                return True, "potential player name pattern", analysis
            
            # 检查是否包含问号（可能是查询）
            if '?' in text or '？' in text:
                if any(word in text_lower for word in ['谁', 'who', '什么', 'what', '哪', 'which']):
                    self.stats['basketball_passed'] += 1
                    return True, "question pattern detected", analysis
            
            # 默认过滤掉
            self.stats['non_basketball_filtered'] += 1
            return False, "no clear basketball indicators", analysis
    
    def _has_person_name_pattern(self, text: str) -> bool:
        """检查是否包含人名模式"""
        # 英文人名模式（首字母大写的词）
        english_name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        if re.search(english_name_pattern, text):
            return True
        
        # 中文人名模式（2-4个中文字符）
        chinese_name_pattern = r'[\u4e00-\u9fff]{2,4}'
        if re.search(chinese_name_pattern, text):
            return True
        
        return False
    
    def batch_filter(self, texts: List[str]) -> List[Tuple[str, bool, str, Dict]]:
        """批量过滤"""
        results = []
        for text in texts:
            is_basketball, reason, analysis = self.is_basketball_domain(text)
            results.append((text, is_basketball, reason, analysis))
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_filtered']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'basketball_rate': self.stats['basketball_passed'] / total * 100,
            'filter_rate': self.stats['non_basketball_filtered'] / total * 100
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_filtered': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0
        }

# 全局实例
basketball_filter = BasketballDomainFilter()
