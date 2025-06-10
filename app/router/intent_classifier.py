"""
BERT意图分类器 - 二分类简单查询和复杂查询
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BERTIntentClassifier:
    """BERT意图分类器"""
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
        # 标签映射
        self.id2label = {0: "simple", 1: "complex"}
        self.label2id = {"simple": 0, "complex": 1}
        
        # 路径配置
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir = Path("results/evaluations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💻 设备: {self.device}")
        logger.info(f"🤖 模型: {self.model_name}")
    
    def load_model(self, model_path: str = None):
        """加载模型"""
        
        if model_path and Path(model_path).exists():
            # 加载微调后的模型
            logger.info(f"📥 加载微调模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # 加载预训练模型
            logger.info(f"📥 加载预训练模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ 模型加载完成")
    
    def load_training_data(self, csv_path: str) -> Tuple[Dataset, Dataset]:
        """加载训练数据"""
        
        logger.info(f"📊 加载训练数据: {csv_path}")
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 数据统计
        logger.info(f"📈 数据统计:")
        logger.info(f"   总数据量: {len(df)}")
        logger.info(f"   Simple查询: {(df['label'] == 0).sum()}")
        logger.info(f"   Complex查询: {(df['label'] == 1).sum()}")
        
        # 划分训练集和测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # 创建Dataset对象
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_texts,
            'labels': test_labels
        })
        
        logger.info(f"📋 数据划分:")
        logger.info(f"   训练集: {len(train_dataset)}")
        logger.info(f"   测试集: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """分词函数"""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {"accuracy": accuracy}
    
    def train(self, csv_path: str, epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """训练模型"""
        
        logger.info("🚀 开始BERT微调训练...")
        
        # 加载预训练模型
        self.load_model()
        
        # 加载训练数据
        train_dataset, test_dataset = self.load_training_data(csv_path)
        
        # 数据预处理
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        # 训练参数
        output_dir = self.model_dir / "bert_intent_classifier"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=str(self.results_dir / "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",  # 禁用自动保存
            load_best_model_at_end=False,  # 禁用最佳模型加载
            save_total_limit=2,
            seed=42
        )
        
        # 训练器
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
        
        # 手动保存最终模型
        final_model_path = self.model_dir / "bert_intent_classifier_final"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        self.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"💾 模型已保存到: {final_model_path}")
        
        # 评估模型
        eval_results = self.evaluate_model(trainer, test_dataset)
        
        return trainer, eval_results
    
    def evaluate_model(self, trainer, test_dataset) -> Dict:
        """详细评估模型"""
        
        logger.info("📊 开始详细评估...")
        
        # 获取预测结果
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 分类报告
        report = classification_report(
            y_true, y_pred,
            target_names=["Simple", "Complex"],
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 详细结果
        evaluation_results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist()
            }
        }
        
        # 打印结果
        logger.info("📈 评估结果:")
        logger.info(f"   准确率: {accuracy:.4f}")
        logger.info("📋 分类报告:")
        print(classification_report(y_true, y_pred, target_names=["Simple", "Complex"]))
        logger.info("🔍 混淆矩阵:")
        print("       Simple  Complex")
        print(f"Simple   {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"Complex  {cm[1][0]:4d}    {cm[1][1]:4d}")
        
        # 保存评估结果
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 评估结果已保存: {results_path}")
        
        return evaluation_results
    
    def classify(self, text: str) -> Tuple[str, float]:
        """分类单个文本"""
        
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()或train()")
        
        # 预处理
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # 解析结果
        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()
        predicted_label = self.id2label[predicted_id]
        
        return predicted_label, confidence
    
    def batch_classify(self, texts: List[str]) -> List[Tuple[str, float]]:
        """批量分类"""
        
        results = []
        for text in texts:
            label, confidence = self.classify(text)
            results.append((label, confidence))
        
        return results
    
    def test_examples(self):
        """测试示例"""
        
        test_cases = [
            "姚明多少岁？",                    # Simple
            "姚明和科比什么关系？",              # Complex
            "科比身高？",                      # Simple
            "比较詹姆斯和科比谁更强？",           # Complex
            "湖人队在哪个城市？",                # Simple
            "分析湖人队的历史成就",               # Complex
        ]
        
        logger.info("🧪 测试示例:")
        for text in test_cases:
            label, confidence = self.classify(text)
            logger.info(f"   '{text}' -> {label} ({confidence:.3f})")

# 全局实例
intent_classifier = BERTIntentClassifier()

if __name__ == "__main__":
    # 训练示例
    csv_path = "data/training/final_training_dataset.csv"
    trainer, results = intent_classifier.train(csv_path, epochs=5)
    intent_classifier.test_examples()
