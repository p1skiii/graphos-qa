#!/usr/bin/env python3
"""
模型评估脚本
评估训练好的BERT模型性能
"""
import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, config_path: str = None):
        """初始化模型评估器"""
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置（如果提供）
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # 获取标签映射
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info(f"✅ 模型加载成功: {self.model_path}")
        logger.info(f"🏷️ 支持的标签: {list(self.label2id.keys())}")
    
    def predict_single(self, text: str):
        """单条文本预测"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        predicted_id = outputs.logits.argmax().item()
        confidence = probabilities[0][predicted_id].item()
        predicted_label = self.id2label[predicted_id]
        
        return predicted_label, confidence, probabilities[0].cpu().numpy()
    
    def predict_batch(self, texts: list):
        """批量预测"""
        predictions = []
        confidences = []
        
        for text in texts:
            pred_label, confidence, _ = self.predict_single(text)
            predictions.append(pred_label)
            confidences.append(confidence)
        
        return predictions, confidences
    
    def evaluate_on_validation_data(self):
        """在验证集上评估"""
        logger.info("📊 在验证集上评估模型...")
        
        # 加载验证数据
        val_path = Path("training/data/processed/val.json")
        if not val_path.exists():
            logger.error("验证数据不存在，请先运行数据预处理")
            return None
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        # 提取文本和真实标签
        texts = [item['text'] for item in val_data]
        true_labels = [item['label'] for item in val_data]
        
        # 批量预测
        pred_labels, confidences = self.predict_batch(texts)
        
        # 计算指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        # 详细报告
        target_names = list(self.label2id.keys())
        report = classification_report(true_labels, pred_labels, target_names=target_names)
        
        # 结果展示
        logger.info("🎯 验证集评估结果:")
        logger.info(f"   准确率: {accuracy:.4f}")
        logger.info(f"   精确率: {precision:.4f}")
        logger.info(f"   召回率: {recall:.4f}")
        logger.info(f"   F1分数: {f1:.4f}")
        logger.info(f"   平均置信度: {np.mean(confidences):.4f}")
        
        logger.info("📋 详细分类报告:")
        logger.info(f"\n{report}")
        
        # 生成混淆矩阵
        self.plot_confusion_matrix(true_labels, pred_labels, target_names)
        
        # 置信度分析
        self.analyze_confidence(pred_labels, true_labels, confidences)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': np.mean(confidences),
            'predictions': pred_labels,
            'true_labels': true_labels,
            'confidences': confidences
        }
    
    def plot_confusion_matrix(self, true_labels, pred_labels, target_names):
        """绘制混淆矩阵"""
        try:
            cm = confusion_matrix(true_labels, pred_labels, labels=target_names)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图片
            output_path = self.model_path.parent / "confusion_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 混淆矩阵保存到: {output_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"⚠️ 无法生成混淆矩阵图片: {e}")
    
    def analyze_confidence(self, pred_labels, true_labels, confidences):
        """分析置信度分布"""
        try:
            # 按正确性分析置信度
            correct_mask = np.array(pred_labels) == np.array(true_labels)
            correct_confidences = np.array(confidences)[correct_mask]
            incorrect_confidences = np.array(confidences)[~correct_mask]
            
            logger.info("🔍 置信度分析:")
            logger.info(f"   正确预测平均置信度: {np.mean(correct_confidences):.4f}")
            logger.info(f"   错误预测平均置信度: {np.mean(incorrect_confidences):.4f}")
            
            # 按标签分析置信度
            logger.info("📊 各标签置信度:")
            for label in set(pred_labels):
                label_mask = np.array(pred_labels) == label
                label_confidences = np.array(confidences)[label_mask]
                logger.info(f"   {label}: {np.mean(label_confidences):.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️ 置信度分析失败: {e}")
    
    def test_custom_examples(self):
        """测试自定义样例"""
        logger.info("🧪 测试自定义样例...")
        
        test_examples = [
            "How old is Kobe Bryant?",
            "Compare LeBron James and Michael Jordan",
            "Did Shaq and Kobe play together?",
            "I love watching basketball games",
            "What's the weather like today?",
            "Who is the tallest player in NBA?",
            "Lakers vs Warriors who is better?",
            "Tell me about basketball history"
        ]
        
        for i, text in enumerate(test_examples, 1):
            pred_label, confidence, probs = self.predict_single(text)
            logger.info(f"样例 {i}: {text}")
            logger.info(f"   预测: {pred_label} (置信度: {confidence:.3f})")
            
            # 显示top-2预测
            top_indices = np.argsort(probs)[-2:][::-1]
            for idx in top_indices:
                label = self.id2label[idx]
                prob = probs[idx]
                logger.info(f"   {label}: {prob:.3f}")
            logger.info("")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python evaluate_model.py <model_path> [config_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(model_path).exists():
        print(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    try:
        evaluator = ModelEvaluator(model_path, config_path)
        
        # 在验证集上评估
        results = evaluator.evaluate_on_validation_data()
        
        # 测试自定义样例
        evaluator.test_custom_examples()
        
        if results:
            print(f"\n🎯 评估完成！")
            print(f"准确率: {results['accuracy']:.4f}")
            print(f"F1分数: {results['f1']:.4f}")
            print(f"平均置信度: {results['avg_confidence']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
