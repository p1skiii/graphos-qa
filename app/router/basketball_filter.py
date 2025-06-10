"""
轻量级篮球领域过滤器 - 仅过滤明显非篮球查询
采用保守策略：只过滤明显不相关的内容，避免误判
"""
import time
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

class BasketballFilter:
    """轻量级篮球领域过滤器"""
    
    def __init__(self):
        # 篮球相关关键词（用于识别篮球内容）
        self.basketball_keywords = {
            # 知名球员（中英文）
            'players': [
                '姚明', '科比', '詹姆斯', '乔丹', '奥尼尔', '邓肯', '诺维茨基', '加内特',
                '韦德', '保罗', '霍华德', '安东尼', '格里芬', '库里', '杜兰特', '威少',
                '哈登', '伦纳德', '字母哥', '浓眉', '恩比德', '约基奇', '东契奇', '塔图姆',
                '布克', '莫兰特', '锡安', '易建联', '王治郅', '巴特尔', '周琦', '郭艾伦',
                '魔术师', '天勾', '大鸟', '石佛', '黑曼巴', '小皇帝', '闪电侠',
                '麦迪', '卡特', '艾弗森', '纳什', '基德', '雷阿伦', '皮尔斯', '加索尔',
                'Yao Ming', 'Kobe', 'LeBron', 'Jordan', 'Shaq', 'Duncan', 'Nowitzki',
                'Garnett', 'Wade', 'Paul', 'Howard', 'Anthony', 'Griffin', 'Curry',
                'Durant', 'Westbrook', 'Harden', 'Leonard', 'Giannis', 'Davis',
                'Embiid', 'Jokic', 'Doncic', 'Tatum', 'Booker', 'Morant', 'Zion'
            ],
            
            # NBA球队（中英文）
            'teams': [
                '湖人', '凯尔特人', '勇士', '火箭', '马刺', '热火', '雷霆', '骑士',
                '公牛', '76人', '快船', '开拓者', '爵士', '掘金', '国王', '太阳',
                '独行侠', '森林狼', '鹈鹕', '魔术', '篮网', '尼克斯', '步行者', '活塞',
                '老鹰', '黄蜂', '奇才', '猛龙', '雄鹿', '公鹿',
                'Lakers', 'Celtics', 'Warriors', 'Rockets', 'Spurs', 'Heat', 'Thunder',
                'Cavaliers', 'Bulls', 'Sixers', 'Clippers', 'Blazers', 'Jazz', 'Nuggets',
                'Kings', 'Suns', 'Mavericks', 'Timberwolves', 'Pelicans', 'Magic',
                'Nets', 'Knicks', 'Pacers', 'Pistons', 'Hawks', 'Hornets', 'Wizards',
                'Raptors', 'Bucks'
            ],
            
            # 篮球专业术语（中英文）
            'terms': [
                '篮球', '扣篮', '三分', '罚球', '篮板', '助攻', '抢断', '盖帽', '失误',
                '得分', '投篮', '命中率', '场均', '总冠军', '季后赛', '常规赛', '全明星',
                'MVP', 'FMVP', '最佳新秀', '最佳防守', '得分王', '篮板王', '助攻王',
                '三双', '四双', '大三元', '空接', '快攻', '挡拆', '突破', '中投',
                '内线', '外线', '后卫', '前锋', '中锋', '控卫', '分卫', '小前锋', '大前锋',
                'NBA', 'CBA', 'FIBA', '选秀', '交易', '自由球员', '球衣退役', '名人堂',
                'basketball', 'dunk', 'three-pointer', 'free throw', 'rebound', 'assist',
                'steal', 'block', 'turnover', 'point', 'shot', 'shooting', 'average',
                'championship', 'playoff', 'regular season', 'all-star', 'triple-double',
                'alley-oop', 'fast break', 'pick and roll', 'drive', 'mid-range',
                'guard', 'forward', 'center', 'point guard', 'shooting guard',
                'small forward', 'power forward', 'draft', 'trade', 'free agent'
            ]
        }
        
        # 明显非篮球的关键词（只包含最明显的非篮球内容）
        self.obvious_non_basketball = {
            # 天气相关
            'weather': ['天气', '下雨', '晴天', '阴天', '雾霾', '温度', '湿度', 'weather', 'rain', 'sunny', 'cloudy', 'temperature'],
            
            # 饮食相关  
            'food': ['菜谱', '做饭', '炒菜', '煮饭', '美食', '餐厅', '饭店', 'recipe', 'cook', 'cooking', 'restaurant', 'food'],
            
            # 交通出行
            'transport': ['地铁', '公交', '打车', '开车', '堵车', '路线', 'subway', 'bus', 'taxi', 'driving', 'traffic'],
            
            # 购物消费
            'shopping': ['购物', '买东西', '商场', '超市', '淘宝', '京东', 'shopping', 'buy', 'store', 'mall'],
            
            # 医疗健康
            'medical': ['医院', '看病', '医生', '药物', '治疗', '症状', 'hospital', 'doctor', 'medicine', 'treatment'],
            
            # 学习工作
            'work_study': ['上班', '工作', '开会', '考试', '作业', '学校', 'work', 'meeting', 'exam', 'homework', 'school'],
            
            # 娱乐影视
            'entertainment': ['电影', '电视剧', '综艺', '动漫', '游戏', 'movie', 'TV show', 'game', 'anime'],
            
            # 技术编程
            'tech': ['编程', '代码', '算法', '数据库', '服务器', 'programming', 'code', 'algorithm', 'database', 'server'],
            
            # 其他明显无关的
            'others': ['政治', '经济', '股票', '房价', 'politics', 'economy', 'stock', 'price']
        }
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0,
            'filter_time': 0.0
        }
    
    def is_basketball_domain(self, user_input: str) -> Tuple[bool, str, Dict]:
        """
        判断是否属于篮球领域
        
        策略：保守过滤，只过滤明显非篮球的内容
        - 如果包含篮球关键词 -> 篮球领域 ✅
        - 如果包含明显非篮球关键词 -> 非篮球领域 ❌
        - 不确定的情况 -> 默认视为篮球领域 ✅ (保守策略)
        
        Returns:
            Tuple[bool, str, Dict]: (是否篮球领域, 原因, 详细分析)
        """
        start_time = time.time()
        
        # 更新统计
        self.stats['total_queries'] += 1
        
        # 基本预处理
        text_lower = user_input.lower().strip()
        
        # 检查长度
        if len(text_lower) < 2:
            self.stats['non_basketball_filtered'] += 1
            processing_time = time.time() - start_time
            self.stats['filter_time'] = (self.stats['filter_time'] * (self.stats['total_queries'] - 1) + processing_time) / self.stats['total_queries']
            return False, "输入太短", {'length': len(text_lower), 'processing_time': processing_time}
        
        # Step 1: 检查篮球关键词
        basketball_matches = []
        for category, keywords in self.basketball_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    basketball_matches.append((category, keyword))
        
        # Step 2: 检查明显非篮球关键词
        non_basketball_matches = []
        for category, keywords in self.obvious_non_basketball.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    non_basketball_matches.append((category, keyword))
        
        # 处理时间
        processing_time = time.time() - start_time
        self.stats['filter_time'] = (self.stats['filter_time'] * (self.stats['total_queries'] - 1) + processing_time) / self.stats['total_queries']
        
        # 详细分析
        analysis = {
            'text_length': len(text_lower),
            'basketball_matches': basketball_matches,
            'non_basketball_matches': non_basketball_matches,
            'basketball_score': len(basketball_matches),
            'non_basketball_score': len(non_basketball_matches),
            'processing_time': processing_time
        }
        
        # 决策逻辑：保守过滤策略
        if non_basketball_matches and not basketball_matches:
            # 明显非篮球内容且无篮球关键词 -> 过滤
            self.stats['non_basketball_filtered'] += 1
            reason = f"明显非篮球: {[match[1] for match in non_basketball_matches[:3]]}"
            return False, reason, analysis
        
        elif basketball_matches:
            # 包含篮球关键词 -> 篮球领域
            self.stats['basketball_passed'] += 1
            reason = f"篮球关键词: {[match[1] for match in basketball_matches[:3]]}"
            return True, reason, analysis
        
        else:
            # 不确定的情况 -> 保守策略，默认通过
            self.stats['basketball_passed'] += 1
            reason = "保守策略：默认通过（无明确非篮球特征）"
            return True, reason, analysis
    
    def batch_filter(self, user_inputs: List[str]) -> List[Tuple[str, bool, str, Dict]]:
        """批量过滤"""
        results = []
        for user_input in user_inputs:
            is_basketball, reason, analysis = self.is_basketball_domain(user_input)
            results.append((user_input, is_basketball, reason, analysis))
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self.stats['total_queries']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'basketball_rate': self.stats['basketball_passed'] / total * 100,
            'filter_rate': self.stats['non_basketball_filtered'] / total * 100,
            'avg_filter_time_ms': self.stats['filter_time'] * 1000
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'basketball_passed': 0,
            'non_basketball_filtered': 0,
            'filter_time': 0.0
        }
        logger.info("🔄 篮球过滤器统计信息已重置")
    
    def get_detailed_report(self) -> str:
        """获取详细报告"""
        stats = self.get_stats()
        total = stats['total_queries']
        
        if total == 0:
            return "📊 篮球过滤器暂无统计数据"
        
        report = f"""
🏀 篮球过滤器报告
{'='*40}

📊 总体统计:
├── 总查询数: {total:,}
├── 篮球领域通过: {stats['basketball_passed']:,} ({stats['basketball_rate']:.1f}%)
├── 非篮球过滤: {stats['non_basketball_filtered']:,} ({stats['filter_rate']:.1f}%)
└── 平均处理时间: {stats['avg_filter_time_ms']:.2f}ms

🎯 策略说明:
├── 保守过滤策略
├── 只过滤明显非篮球内容
├── 不确定的情况默认通过
└── 避免误杀篮球相关查询

⚡ 性能指标:
└── 超轻量级: < 1ms 处理时间
        """
        
        return report

# 全局过滤器实例
basketball_filter = BasketballFilter()

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 篮球相关 - 应该通过
        "姚明多少岁？",
        "科比身高是多少？", 
        "湖人队在哪个城市？",
        "詹姆斯和韦德什么关系？",
        "Kobe how tall?",
        "Lakers vs Celtics",
        
        # 明显非篮球 - 应该过滤
        "今天天气怎么样？",
        "附近有什么好吃的餐厅？",
        "怎么去地铁站？",
        "股票价格如何？",
        "what's the weather like?",
        "how to cook pasta?",
        
        # 不确定的情况 - 保守策略，应该通过
        "张三是谁？",
        "什么是三角进攻？",
        "比较一下两个人",
        "who is better?",
        "最高的人是谁？"
    ]
    
    print("🏀 轻量级篮球过滤器测试:")
    print("="*50)
    
    for query in test_cases:
        is_basketball, reason, analysis = basketball_filter.is_basketball_domain(query)
        status = "✅ 通过" if is_basketball else "❌ 过滤"
        print(f"{status} | {query}")
        print(f"   原因: {reason}")
        print(f"   时间: {analysis['processing_time']*1000:.2f}ms")
        print("-" * 30)
    
    print(basketball_filter.get_detailed_report())
