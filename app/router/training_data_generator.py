"""
训练数据生成器 - 生成黄金数据集和伪标签数据
"""
import csv
import json
import random
import re
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(self):
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_golden_dataset(self) -> List[Dict]:
        """创建黄金数据集 - 高质量手工标注数据"""
        
        logger.info("🏆 创建黄金数据集...")
        
        # 🔵 Simple Query (简单查询) - 标签: 0
        # 特征: 单一信息点，直接查询，明确答案
        simple_queries = [
            # === 年龄相关 ===
            "姚明多少岁？", "科比几岁？", "詹姆斯今年多大？", "邓肯的年龄？",
            "姚明什么时候出生的？", "科比哪年出生？", "詹姆斯生日？",
            "Yao Ming how old?", "Kobe age?", "LeBron age?", "Duncan age?",
            
            # === 身高体重 ===
            "姚明多高？", "科比身高？", "詹姆斯有多高？", "奥尼尔多高？",
            "姚明体重？", "科比多重？", "詹姆斯体重？", "邓肯多重？",
            "Yao Ming height?", "Kobe height?", "LeBron height?", "Shaq height?",
            
            # === 球队信息 ===
            "姚明在哪个队？", "科比效力哪个球队？", "詹姆斯现在哪个队？",
            "湖人队在哪个城市？", "火箭队主场在哪？", "马刺队在哪？",
            "Yao Ming which team?", "Kobe team?", "LeBron team?", "Lakers city?",
            
            # === 位置信息 ===
            "姚明什么位置？", "科比打什么位置？", "詹姆斯位置？",
            "Yao Ming position?", "Kobe position?", "LeBron position?",
            
            # === 统计数据 ===
            "科比总得分？", "詹姆斯生涯得分？", "姚明场均得分？",
            "科比几个冠军？", "詹姆斯几个总冠军？", "邓肯几个戒指？",
            "Kobe total points?", "LeBron career points?", "Duncan championships?",
            
            # === 球衣号码 ===
            "科比几号球衣？", "詹姆斯球衣号码？", "姚明几号？",
            "Kobe jersey number?", "LeBron number?", "Yao Ming number?",
            
            # === 退役信息 ===
            "科比退役了吗？", "姚明什么时候退役？", "邓肯还在打球吗？",
            "Kobe retired?", "when Yao Ming retire?", "Duncan still playing?",
            
            # === 国籍信息 ===
            "姚明哪国人？", "科比来自哪里？", "詹姆斯国籍？",
            "Yao Ming from which country?", "Kobe nationality?", "LeBron from where?",
            
            # === 选秀信息 ===
            "詹姆斯第几顺位？", "科比几号秀？", "姚明选秀顺位？",
            "LeBron draft position?", "Kobe draft pick?", "Yao Ming draft?"
        ]
        
        # 🔴 Complex Query (复杂查询) - 标签: 1
        # 特征: 多信息点，关系推理，比较分析，聚合统计
        complex_queries = [
            # === 关系推理 ===
            "姚明和科比什么关系？", "詹姆斯和韦德认识吗？", "邓肯和帕克是队友吗？",
            "科比有哪些队友？", "姚明的队友都有谁？", "詹姆斯和谁一起打过球？",
            "Yao Ming and Kobe relationship?", "LeBron and Wade teammates?",
            "who are Kobe teammates?", "Yao Ming teammates list?",
            
            # === 比较分析 ===
            "姚明和奥尼尔谁更高？", "科比和詹姆斯谁更强？", "湖人和凯尔特人哪个更厉害？",
            "比较邓肯和加内特", "东部西部哪个更强？", "分析科比和乔丹的区别",
            "Yao Ming vs Shaq who taller?", "Kobe vs LeBron who better?",
            "compare Duncan and Garnett", "Lakers vs Celtics which better?",
            
            # === 路径推理 ===
            "通过什么路径连接姚明和科比？", "姚明和詹姆斯有什么共同点？",
            "从火箭队到湖人队有哪些球员转会？", "中国球员和美国球员的联系？",
            "how connect Yao Ming and Kobe?", "path from Rockets to Lakers?",
            
            # === 聚合统计 ===
            "湖人队总共多少球员？", "NBA历史最高球员是谁？", "哪个球队冠军最多？",
            "火箭队平均年龄？", "得分最多的球员？", "最年轻的MVP？",
            "how many Lakers players?", "tallest NBA player ever?", "most championships team?",
            
            # === 分析解释 ===
            "为什么科比叫黑曼巴？", "姚明对中国篮球的影响？", "分析湖人王朝历史",
            "解释什么是三角进攻？", "如何评价詹姆斯生涯？", "马刺文化是什么？",
            "why Kobe called Black Mamba?", "Yao Ming impact on basketball?",
            "analyze Lakers dynasty", "explain triangle offense?",
            
            # === 多条件查询 ===
            "既是湖人球员又拿过冠军的有谁？", "和姚明同时期超过2米的中锋？",
            "科比职业生涯所有队友", "在火箭打过球的中国球员",
            "Lakers players who won championship?", "centers over 7 feet in Yao Ming era?",
            
            # === 时间序列 ===
            "姚明职业生涯发展轨迹", "科比从新秀到退役历程", "湖人阵容变化",
            "Yao Ming career timeline", "Kobe development journey", "Lakers roster changes",
            
            # === 假设推理 ===
            "如果姚明没受伤会怎样？", "科比和詹姆斯同队会如何？",
            "假设乔丹在现代会怎样？", "如果邓肯去了其他队？",
            "what if Yao Ming no injury?", "if Kobe and LeBron same team?",
            
            # === 深度分析 ===
            "分析中国球员在NBA的影响", "探讨篮球运动的全球化", "NBA选秀制度演变",
            "analyze Chinese players impact in NBA", "discuss basketball globalization"
        ]
        
        # 构建数据集
        golden_dataset = []
        
        for query in simple_queries:
            golden_dataset.append({
                'text': query,
                'label': 0,
                'label_name': 'simple_query',
                'source': 'golden',
                'quality': 'high'
            })
        
        for query in complex_queries:
            golden_dataset.append({
                'text': query,
                'label': 1,
                'label_name': 'complex_query',
                'source': 'golden',
                'quality': 'high'
            })
        
        logger.info(f"✅ 黄金数据集创建完成:")
        logger.info(f"   Simple Query: {len(simple_queries)} 条")
        logger.info(f"   Complex Query: {len(complex_queries)} 条")
        logger.info(f"   总计: {len(golden_dataset)} 条")
        
        return golden_dataset
    
    def generate_pseudo_labeled_data(self, count: int = 500) -> List[Dict]:
        """生成伪标签数据"""
        
        logger.info(f"🤖 生成伪标签数据 {count} 条...")
        
        # 用于生成的模板
        simple_templates = [
            "{player}多少岁？", "{player}身高？", "{player}在哪队？",
            "{player}什么位置？", "{player}几号球衣？", "{team}在哪个城市？",
            "{player} age?", "{player} height?", "{player} team?",
            "{player} position?", "{team} location?"
        ]
        
        complex_templates = [
            "{player1}和{player2}什么关系？", "{player1}和{player2}谁更强？",
            "哪些球员是{player}的队友？", "分析{team}的历史", "为什么{player}被称为{nickname}？",
            "{player1} and {player2} relationship?", "who better {player1} or {player2}?",
            "analyze {team} history", "teammates of {player}?"
        ]
        
        # 实体库
        players = ['姚明', '科比', '詹姆斯', '乔丹', '邓肯', '奥尼尔', '库里', '杜兰特',
                  'Yao Ming', 'Kobe', 'LeBron', 'Jordan', 'Duncan', 'Shaq', 'Curry']
        teams = ['湖人', '火箭', '马刺', '勇士', '凯尔特人', 
                'Lakers', 'Rockets', 'Spurs', 'Warriors', 'Celtics']
        nicknames = ['黑曼巴', '小皇帝', '大鲨鱼', '石佛', 'Black Mamba', 'King James']
        
        pseudo_data = []
        
        for _ in range(count):
            # 随机选择类别
            if random.random() < 0.5:  # 50% simple, 50% complex
                label = 0
                template = random.choice(simple_templates)
            else:
                label = 1
                template = random.choice(complex_templates)
            
            # 填充模板
            try:
                if '{player1}' in template and '{player2}' in template:
                    selected_players = random.sample(players, 2)
                    text = template.format(player1=selected_players[0], player2=selected_players[1])
                elif '{player}' in template:
                    if '{nickname}' in template:
                        text = template.format(
                            player=random.choice(players),
                            nickname=random.choice(nicknames)
                        )
                    else:
                        text = template.format(player=random.choice(players))
                elif '{team}' in template:
                    text = template.format(team=random.choice(teams))
                else:
                    text = template
                
                pseudo_data.append({
                    'text': text,
                    'label': label,
                    'label_name': 'simple_query' if label == 0 else 'complex_query',
                    'source': 'pseudo',
                    'quality': 'medium'
                })
            except:
                continue
        
        logger.info(f"✅ 伪标签数据生成完成: {len(pseudo_data)} 条")
        return pseudo_data
    
    def validate_and_correct_pseudo_data(self, pseudo_data: List[Dict]) -> List[Dict]:
        """验证和修正伪标签数据"""
        
        logger.info("🔍 验证和修正伪标签数据...")
        
        validated_data = []
        corrections = 0
        
        for item in pseudo_data:
            text = item['text'].lower()
            original_label = item['label']
            
            # 简单规则验证
            corrected_label = self._rule_based_label_correction(text)
            
            if corrected_label is not None and corrected_label != original_label:
                item['label'] = corrected_label
                item['label_name'] = 'simple_query' if corrected_label == 0 else 'complex_query'
                item['corrected'] = True
                corrections += 1
            
            validated_data.append(item)
        
        logger.info(f"✅ 数据验证完成，修正了 {corrections} 个标签")
        return validated_data
    
    def _rule_based_label_correction(self, text: str) -> int:
        """基于规则的标签修正"""
        
        # Simple query 模式
        simple_patterns = [
            r'(多少岁|几岁|age|how old)',
            r'(多高|身高|height|how tall)',
            r'(多重|体重|weight|how heavy)',
            r'(在哪|which team|what team)',
            r'(几号|number|jersey)',
            r'(退役|retired|retire)',
            r'(国籍|nationality|from where)'
        ]
        
        # Complex query 模式
        complex_patterns = [
            r'(什么关系|relationship|认识|know)',
            r'(谁更|who better|比较|compare)',
            r'(队友|teammate|一起|together)',
            r'(分析|analyze|为什么|why)',
            r'(如果|假设|what if|suppose)',
            r'(哪些|who are|所有|all)',
            r'(路径|path|连接|connect)'
        ]
        
        # 检查简单查询模式
        for pattern in simple_patterns:
            if re.search(pattern, text):
                return 0
        
        # 检查复杂查询模式
        for pattern in complex_patterns:
            if re.search(pattern, text):
                return 1
        
        return None  # 无法确定
    
    def combine_datasets(self, golden_data: List[Dict], pseudo_data: List[Dict]) -> List[Dict]:
        """合并黄金数据集和伪标签数据"""
        
        logger.info("🔗 合并数据集...")
        
        combined_data = golden_data + pseudo_data
        
        # 打乱数据
        random.shuffle(combined_data)
        
        # 统计信息
        total = len(combined_data)
        simple_count = sum(1 for item in combined_data if item['label'] == 0)
        complex_count = sum(1 for item in combined_data if item['label'] == 1)
        golden_count = sum(1 for item in combined_data if item['source'] == 'golden')
        pseudo_count = sum(1 for item in combined_data if item['source'] == 'pseudo')
        
        logger.info(f"✅ 数据集合并完成:")
        logger.info(f"   总计: {total} 条")
        logger.info(f"   Simple: {simple_count} 条 ({simple_count/total*100:.1f}%)")
        logger.info(f"   Complex: {complex_count} 条 ({complex_count/total*100:.1f}%)")
        logger.info(f"   黄金数据: {golden_count} 条 ({golden_count/total*100:.1f}%)")
        logger.info(f"   伪标签数据: {pseudo_count} 条 ({pseudo_count/total*100:.1f}%)")
        
        return combined_data
    
    def save_training_data(self, data: List[Dict], filename: str = "training_dataset"):
        """保存训练数据"""
        
        # 保存为CSV格式（用于模型训练）
        csv_path = self.data_dir / f"{filename}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label'])
            writer.writeheader()
            for item in data:
                writer.writerow({
                    'text': item['text'],
                    'label': item['label']
                })
        
        # 保存为JSON格式（包含详细信息）
        json_path = self.data_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 训练数据已保存:")
        logger.info(f"   CSV: {csv_path}")
        logger.info(f"   JSON: {json_path}")
        
        return csv_path, json_path
    
    def generate_complete_dataset(self) -> Tuple[str, str]:
        """生成完整的训练数据集"""
        
        logger.info("🚀 开始生成完整训练数据集...")
        
        # 1. 创建黄金数据集
        golden_data = self.create_golden_dataset()
        
        # 2. 生成伪标签数据
        pseudo_data = self.generate_pseudo_labeled_data(count=300)
        
        # 3. 验证和修正伪标签数据
        validated_pseudo_data = self.validate_and_correct_pseudo_data(pseudo_data)
        
        # 4. 合并数据集
        final_dataset = self.combine_datasets(golden_data, validated_pseudo_data)
        
        # 5. 保存训练数据
        csv_path, json_path = self.save_training_data(final_dataset, "final_training_dataset")
        
        logger.info("✅ 完整训练数据集生成完成！")
        
        return str(csv_path), str(json_path)

# 全局实例
training_data_generator = TrainingDataGenerator()

if __name__ == "__main__":
    # 生成训练数据
    csv_path, json_path = training_data_generator.generate_complete_dataset()
