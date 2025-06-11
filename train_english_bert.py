#!/usr/bin/env python3
"""
英文BERT意图分类器训练脚本
使用本地中文BERT模型作为基础，训练英文5类数据
"""

import torch
import pandas as pd
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnglishBERTTrainer:
    """英文BERT训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 5类意图标签映射
        self.id2label = {
            0: "ATTRIBUTE_QUERY",        # 属性查询
            1: "SIMPLE_RELATION_QUERY",  # 简单关系查询  
            2: "COMPLEX_RELATION_QUERY", # 复杂关系查询
            3: "COMPARATIVE_QUERY",      # 比较查询
            4: "DOMAIN_CHITCHAT"         # 领域内闲聊
        }
        self.label2id = {
            "ATTRIBUTE_QUERY": 0,
            "SIMPLE_RELATION_QUERY": 1, 
            "COMPLEX_RELATION_QUERY": 2,
            "COMPARATIVE_QUERY": 3,
            "DOMAIN_CHITCHAT": 4
        }
        
        logger.info(f"💻 设备: {self.device}")
        
    def load_local_model(self, model_path="models/bert_intent_classifier_final"):
        """加载本地模型作为基础"""
        
        logger.info(f"📥 加载本地模型: {model_path}")
        
        # 加载tokenizer（保持中文tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载config并修改为5类
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 5
        config.id2label = self.id2label
        config.label2id = self.label2id
        
        # 重新初始化模型（分类头会被重新初始化）
        self.model = AutoModelForSequenceClassification.from_config(config)
        
        # 加载预训练权重（除了分类头）
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # 复制除分类头以外的所有权重
        model_dict = self.model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # 过滤掉分类头的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        self.model.to(self.device)
        logger.info("✅ 模型加载完成，分类头已重新初始化为5类")
        
    def tokenize_function(self, examples):
        """分词函数"""
        return self.tokenizer(
            examples['text'], 
            truncation=True, 
            padding=True, 
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
        }
    
    def load_data(self, csv_path):
        """加载训练数据"""
        
        logger.info(f"📊 加载训练数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 数据统计
        logger.info(f"📈 数据统计:")
        logger.info(f"   总数据量: {len(df)}")
        
        for label_id, label_name in self.id2label.items():
            count = (df['label'] == label_id).sum()
            logger.info(f"   {label_name}: {count} samples")
        
        # 划分训练集和测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # 创建Dataset
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels  # 注意这里是'labels'不是'label'
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'labels': test_labels
        })
        
        # 分词处理
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        return train_dataset, test_dataset
    
    def train(self, csv_path, epochs=3, batch_size=16):
        """训练模型"""
        
        # 加载模型
        self.load_local_model()
        
        # 加载数据
        train_dataset, test_dataset = self.load_data(csv_path)
        
        # 创建输出目录
        output_dir = Path("models/bert_english_5class")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="results/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            seed=42
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 开始训练
        logger.info("⏳ 开始训练...")
        trainer.train()
        
        # 保存最终模型
        final_model_path = Path("models/bert_english_5class_final")
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"💾 模型已保存到: {final_model_path}")
        
        # 评估
        eval_results = trainer.evaluate()
        logger.info(f"📊 最终评估结果: {eval_results}")
        
        # 保存评估结果
        results_file = Path("results/evaluations/english_5class_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return trainer, eval_results
    
    def test_examples(self):
        """测试示例"""
        
        test_cases = [
            "How old is Yao Ming?",                           # ATTRIBUTE_QUERY
            "Which team does LeBron James play for?",         # SIMPLE_RELATION_QUERY  
            "What's Kobe's height?",                          # ATTRIBUTE_QUERY
            "Who is better, LeBron James or Kobe Bryant?",    # COMPARATIVE_QUERY
            "Which city are the Lakers located in?",         # ATTRIBUTE_QUERY
            "Analyze the Lakers' historical achievements",    # COMPLEX_RELATION_QUERY
            "What do you think about basketball?",            # DOMAIN_CHITCHAT
            "List all MVP players who played with Shaq",     # COMPLEX_RELATION_QUERY
        ]
        
        logger.info("🧪 测试示例:")
        
        for text in test_cases:
            # 分词
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()
                predicted_label = self.id2label[predicted_id]
                
                logger.info(f"   '{text}' -> {predicted_label} ({confidence:.3f})")

if __name__ == "__main__":
    trainer = EnglishBERTTrainer()
    
    # 训练
    csv_path = "data/training/english_fine_grained_training_dataset.csv"
    model_trainer, results = trainer.train(csv_path, epochs=5)
    
    # 测试
    trainer.test_examples()
