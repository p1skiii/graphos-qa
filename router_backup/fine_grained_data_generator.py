"""
Fine-Grained English Training Data Generator - Support 5-class Intent Classification
Generate training data for Tier 2 (BERT Diagnostic Center) of the three-tier routing architecture
All samples generated in English to avoid semantic confusion
"""
import pandas as pd
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class EnglishFineGrainedDataGenerator:
    """Fine-grained English training data generator"""
    
    def __init__(self):
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 5-class intent classification label definitions
        self.intent_labels = {
            "ATTRIBUTE_QUERY": 0,        # Attribute queries
            "SIMPLE_RELATION_QUERY": 1,  # Simple relation queries
            "COMPLEX_RELATION_QUERY": 2, # Complex relation queries
            "COMPARATIVE_QUERY": 3,      # Comparative queries
            "DOMAIN_CHITCHAT": 4         # Domain-specific chitchat
        }
        
        # Basketball entity vocabulary (English)
        self.entities = {
            'players': [
                'Yao Ming', 'Kobe Bryant', 'Michael Jordan', 'LeBron James', 'Stephen Curry', 'Kevin Durant', 
                'Dwyane Wade', 'Tim Duncan', 'Shaquille O\'Neal', 'Vince Carter', 'Tracy McGrady', 
                'Allen Iverson', 'Steve Nash', 'Dirk Nowitzki', 'Kevin Garnett', 'Yi Jianlian', 
                'Zhizhi Wang', 'Mengke Bateer', 'Jeremy Lin', 'Scottie Pippen', 'Magic Johnson',
                'Larry Bird', 'Kareem Abdul-Jabbar', 'Bill Russell', 'Wilt Chamberlain'
            ],
            'teams': [
                'Lakers', 'Warriors', 'Bulls', 'Celtics', 'Heat', 'Spurs', 'Thunder', 'Rockets',
                'Nets', '76ers', 'Knicks', 'Clippers', 'Kings', 'Suns', 'Nuggets', 'Jazz',
                'Trail Blazers', 'Timberwolves', 'Pelicans', 'Mavericks', 'Pacers', 'Bucks', 
                'Cavaliers', 'Pistons', 'Magic', 'Hawks', 'Hornets', 'Wizards', 'Raptors'
            ],
            'attributes': [
                'height', 'weight', 'age', 'position', 'jersey number', 'birthday', 'nationality', 
                'draft year', 'home court', 'founding year', 'coach', 'owner'
            ],
            'achievements': [
                'MVP', 'championship', 'scoring champion', 'rebounding leader', 'assist leader', 
                'Defensive Player of the Year', 'Sixth Man of the Year', 'Rookie of the Year', 
                'All-Star', 'Hall of Fame'
            ]
        }
    
    def generate_attribute_queries(self, count: int = 100) -> List[Dict]:
        """Generate attribute query data"""
        
        templates = [
            "How tall is {player}?",
            "What is {player}'s height?", 
            "How old is {player}?",
            "What jersey number does {player} wear?",
            "What position does {player} play?",
            "When was {player} born?",
            "Where is {team}'s home court?",
            "When was {team} founded?",
            "Who is {team}'s current coach?",
            "What are {player}'s basic stats?",
            "Tell me about {player}'s career data"
        ]
        
        data = []
        for _ in range(count):
            template = random.choice(templates)
            if '{player}' in template:
                entity = random.choice(self.entities['players'])
                text = template.format(player=entity)
            else:
                entity = random.choice(self.entities['teams'])  
                text = template.format(team=entity)
            
            data.append({
                'text': text,
                'label': self.intent_labels['ATTRIBUTE_QUERY'],
                'intent': 'ATTRIBUTE_QUERY'
            })
        
        return data
    
    def generate_simple_relation_queries(self, count: int = 100) -> List[Dict]:
        """Generate simple relation query data"""
        
        templates = [
            "Who are {player}'s teammates?",
            "Who is {player}'s coach?", 
            "Which team does {player} play for?",
            "Who does {player} play with?",
            "Which players are on {team}?",
            "Who are {team}'s star players?",
            "Who is {player}'s agent?",
            "Who owns {team}?",
            "Which coach did {player} learn from?",
            "Who is {team}'s captain?"
        ]
        
        data = []
        for _ in range(count):
            template = random.choice(templates)
            if '{player}' in template:
                entity = random.choice(self.entities['players'])
                text = template.format(player=entity)
            else:
                entity = random.choice(self.entities['teams'])
                text = template.format(team=entity)
                
            data.append({
                'text': text,
                'label': self.intent_labels['SIMPLE_RELATION_QUERY'],
                'intent': 'SIMPLE_RELATION_QUERY'
            })
        
        return data
    
    def generate_complex_relation_queries(self, count: int = 100) -> List[Dict]:
        """Generate complex relation query data"""
        
        templates = [
            "Which players were teammates with {player1} and also won championships?",
            "Which players played for both {team1} and {team2}?",
            "Analyze {player}'s impact on {team}'s historical status",
            "Which coaches did {player} work with throughout their career?",
            "List all MVP players who had interactions with {player}", 
            "Review {team}'s important trades and transfers in history",
            "Analyze {player}'s career trajectory and key milestones",
            "Explore the mentorship relationship between {player1} and {player2}",
            "Summarize {team}'s core roster changes during their dynasty period",
            "Analyze the evolution of modern basketball tactical systems"
        ]
        
        data = []
        for _ in range(count):
            template = random.choice(templates)
            
            # Randomly fill entities
            entities_in_template = []
            if '{player1}' in template:
                entities_in_template.append(('player1', random.choice(self.entities['players'])))
            if '{player2}' in template:  
                entities_in_template.append(('player2', random.choice(self.entities['players'])))
            if '{player}' in template:
                entities_in_template.append(('player', random.choice(self.entities['players'])))
            if '{team1}' in template:
                entities_in_template.append(('team1', random.choice(self.entities['teams'])))
            if '{team2}' in template:
                entities_in_template.append(('team2', random.choice(self.entities['teams'])))
            if '{team}' in template:
                entities_in_template.append(('team', random.choice(self.entities['teams'])))
            
            text = template
            for placeholder, entity in entities_in_template:
                text = text.replace(f'{{{placeholder}}}', entity)
                
            data.append({
                'text': text,
                'label': self.intent_labels['COMPLEX_RELATION_QUERY'],
                'intent': 'COMPLEX_RELATION_QUERY'
            })
        
        return data
    
    def generate_comparative_queries(self, count: int = 100) -> List[Dict]:
        """Generate comparative query data"""
        
        templates = [
            "Compare {player1} and {player2}'s career achievements",
            "Who has better scoring ability, {player1} or {player2}?",
            "Compare {team1} and {team2}'s historical records",
            "What are the differences in technical styles between {player1} and {player2}?",
            "Analyze the leadership differences between {player1} and {player2}",
            "Which team has more championship potential, {team1} or {team2}?",
            "Evaluate the historical status of {player1} vs {player2}",
            "Compare and analyze {player1} and {player2}'s playing styles",
            "{team1} vs {team2}, which team is stronger?",
            "Compare {player1} and {player2}'s influence on basketball"
        ]
        
        data = []
        for _ in range(count):
            template = random.choice(templates)
            
            if 'team1' in template:
                team1 = random.choice(self.entities['teams'])
                team2 = random.choice([t for t in self.entities['teams'] if t != team1])
                text = template.format(team1=team1, team2=team2)
            else:
                player1 = random.choice(self.entities['players'])
                player2 = random.choice([p for p in self.entities['players'] if p != player1])
                text = template.format(player1=player1, player2=player2)
                
            data.append({
                'text': text,
                'label': self.intent_labels['COMPARATIVE_QUERY'],
                'intent': 'COMPARATIVE_QUERY'
            })
        
        return data
    
    def generate_domain_chitchat(self, count: int = 100) -> List[Dict]:
        """Generate domain-specific chitchat data"""
        
        templates = [
            "Who do you think is the GOAT?",
            "What makes basketball so appealing?",
            "Is the NBA still entertaining nowadays?",
            "What are your thoughts on basketball?",
            "Share your favorite basketball moments",
            "What has basketball brought to you?",
            "How do you evaluate modern basketball development?",
            "What do you think is the spirit of basketball?",
            "Share your basketball story",
            "How has basketball changed things?",
            "Talk about basketball culture's influence",
            "What do you think about basketball commercialization?",
            "What attracts you most about basketball?",
            "Share the inspiration basketball brings to people",
            "What does a basketball hero look like in your mind?"
        ]
        
        data = []
        for _ in range(count):
            text = random.choice(templates)
            data.append({
                'text': text,
                'label': self.intent_labels['DOMAIN_CHITCHAT'],
                'intent': 'DOMAIN_CHITCHAT'
            })
        
        return data
    
    def generate_complete_dataset(self, total_samples: int = 500) -> pd.DataFrame:
        """Generate complete 5-class training dataset"""
        
        # Allocation ratio for each class
        samples_per_class = total_samples // 5
        
        logger.info(f"🎯 Starting to generate 5-class fine-grained training data...")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Samples per class: {samples_per_class}")
        
        all_data = []
        
        # Generate data for each class
        all_data.extend(self.generate_attribute_queries(samples_per_class))
        all_data.extend(self.generate_simple_relation_queries(samples_per_class))
        all_data.extend(self.generate_complex_relation_queries(samples_per_class))
        all_data.extend(self.generate_comparative_queries(samples_per_class))
        all_data.extend(self.generate_domain_chitchat(samples_per_class))
        
        # Shuffle data order
        random.shuffle(all_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Data statistics
        logger.info("📊 Data generation completed:")
        for intent, label in self.intent_labels.items():
            count = len(df[df['label'] == label])
            percentage = count / len(df) * 100
            logger.info(f"   {intent}: {count} samples ({percentage:.1f}%)")
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, filename: str = "english_fine_grained_training_dataset"):
        """Save training data"""
        
        # Save CSV format
        csv_path = self.data_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"💾 CSV data saved: {csv_path}")
        
        # Save JSON format  
        json_path = self.data_dir / f"{filename}.json"
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"💾 JSON data saved: {json_path}")
        
        return csv_path, json_path

def generate_english_training_data(total_samples: int = 500):
    """Main function to generate English fine-grained training data"""
    
    generator = EnglishFineGrainedDataGenerator()
    
    # Generate dataset
    df = generator.generate_complete_dataset(total_samples)
    
    # Save data
    csv_path, json_path = generator.save_training_data(df)
    
    logger.info("✅ English fine-grained training data generation completed!")
    return df, csv_path, json_path

if __name__ == "__main__":
    # Generate 500 samples of 5-class English training data
    df, csv_path, json_path = generate_english_training_data(500)
    
    print("🎯 English Fine-grained Training Data Generation Completed!")
    print(f"📊 Total samples: {len(df)}")
    print(f"💾 Files saved:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
