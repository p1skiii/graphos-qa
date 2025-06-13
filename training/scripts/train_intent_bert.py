#!/usr/bin/env python3
"""
BERT意图分类训练脚本
独立的训练模块，与业务代码分离
"""
import os
import sys
import yaml
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntentBERTTrainer:
    def __init__(self, config_path: str):
        """初始化BERT训练器"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.output_dir = Path(self.config['output']['model_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 使用设备: {self.device}")
        
        # 标签映射
        self.id2label = {v: k for k, v in self.config['labels'].items()}
        self.label2id = self.config['labels']
        
    def load_data(self):
        """加载处理后的数据"""
        logger.info("📂 加载训练数据...")
        
        # 检查数据文件是否存在
        train_path = Path("training/data/processed/train.json")
        val_path = Path("training/data/processed/val.json")
        
        if not train_path.exists() or not val_path.exists():
            logger.error("❌ 处理后的数据文件不存在，请先运行数据预处理脚本")
            raise FileNotFoundError("请先运行: python prepare_data.py")
        
        # 加载JSON数据
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        # 转换为Dataset格式
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        logger.info(f"📊 训练集: {len(train_dataset)} 样本")
        logger.info(f"📊 验证集: {len(val_dataset)} 样本")
        
        return train_dataset, val_dataset
    
    def tokenize_data(self, train_dataset: Dataset, val_dataset: Dataset):
        """分词处理"""
        logger.info("🔤 开始分词处理...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            """分词函数"""
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,  # 使用DataCollator进行动态padding
                max_length=self.config['model']['max_length']
            )
        
        # 应用分词
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # 重命名标签列
        train_dataset = train_dataset.rename_column("label_id", "labels")
        val_dataset = val_dataset.rename_column("label_id", "labels")
        
        # 设置格式
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        logger.info("✅ 分词处理完成")
        
        return train_dataset, val_dataset, tokenizer
    
    def create_model(self):
        """创建模型"""
        logger.info(f"🤖 创建模型: {self.model_name}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.config['model']['num_labels'],
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        return model
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # 准确率
        accuracy = accuracy_score(labels, predictions)
        
        # 精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self):
        """执行训练"""
        logger.info("🚀 开始BERT训练...")
        
        try:
            # 1. 加载数据
            train_dataset, val_dataset = self.load_data()
            
            # 2. 分词处理
            train_dataset, val_dataset, tokenizer = self.tokenize_data(train_dataset, val_dataset)
            
            # 3. 创建模型
            model = self.create_model()
            
            # 4. 数据整理器
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # 5. 训练参数
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=self.config['training']['num_epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                per_device_eval_batch_size=self.config['training']['batch_size'],
                learning_rate=self.config['training']['learning_rate'],
                warmup_steps=self.config['training']['warmup_steps'],
                weight_decay=self.config['training']['weight_decay'],
                logging_dir=self.config['output']['logs_dir'],
                logging_steps=self.config['output']['logging_steps'],
                evaluation_strategy="steps",
                eval_steps=self.config['output']['eval_steps'],
                save_steps=self.config['output']['save_steps'],
                load_best_model_at_end=True,
                metric_for_best_model=self.config['early_stopping']['metric'],
                greater_is_better=True,
                save_total_limit=3,  # 只保留最近3个checkpoint
                fp16=self.config['training']['fp16'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                report_to="none",  # 禁用wandb等报告
            )
            
            # 6. 早停回调
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config['early_stopping']['patience']
            )
            
            # 7. 训练器
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[early_stopping]
            )
            
            # 8. 开始训练
            logger.info("🎯 开始训练...")
            trainer.train()
            
            # 9. 保存最终模型
            final_model_path = self.output_dir / "final"
            trainer.save_model(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            
            # 10. 评估模型
            logger.info("📊 评估模型...")
            eval_results = trainer.evaluate()
            
            logger.info("🎯 训练完成！评估结果:")
            for key, value in eval_results.items():
                logger.info(f"   {key}: {value:.4f}")
            
            # 11. 生成详细的分类报告
            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            target_names = [self.id2label[i] for i in range(len(self.id2label))]
            report = classification_report(y_true, y_pred, target_names=target_names)
            
            logger.info("📋 详细分类报告:")
            logger.info(f"\n{report}")
            
            # 保存报告
            with open(self.output_dir / "classification_report.txt", 'w') as f:
                f.write(report)
            
            logger.info(f"✅ 模型训练完成！保存位置: {final_model_path}")
            
            return str(final_model_path)
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            raise

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python train_intent_bert.py <bert_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    trainer = IntentBERTTrainer(config_path)
    model_path = trainer.train()
    
    print(f"\n🎉 训练完成！")
    print(f"📁 模型保存路径: {model_path}")
    print(f"🔗 你可以在业务代码中使用该模型进行意图分类")

if __name__ == "__main__":
    main()
