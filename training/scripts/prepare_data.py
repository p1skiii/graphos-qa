#!/usr/bin/env python3
"""
数据预处理脚本
将原始CSV数据转换为训练格式
"""
import os
import sys
import yaml
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_config_path: str, bert_config_path: str):
        """初始化数据预处理器"""
        with open(data_config_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        with open(bert_config_path, 'r') as f:
            self.bert_config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.data_config['data_sources']['raw_data_path'])
        self.processed_train_path = Path(self.data_config['data_sources']['processed_train_path'])
        self.processed_val_path = Path(self.data_config['data_sources']['processed_val_path'])
        
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        logger.info(f"📂 加载原始数据: {self.raw_data_path}")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"原始数据文件不存在: {self.raw_data_path}")
        
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"📊 原始数据形状: {df.shape}")
        logger.info(f"📊 数据列: {list(df.columns)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        logger.info("🧹 开始数据清洗...")
        
        # 获取配置
        text_col = self.data_config['field_mapping']['text_column']
        label_col = self.data_config['field_mapping']['label_column']
        preprocessing = self.data_config['preprocessing']
        
        initial_size = len(df)
        
        # 删除空值
        df = df.dropna(subset=[text_col, label_col])
        logger.info(f"📊 删除空值后: {len(df)} 行 (删除了 {initial_size - len(df)} 行)")
        
        # 文本长度过滤
        min_len = preprocessing['min_text_length']
        max_len = preprocessing['max_text_length']
        df = df[df[text_col].str.len().between(min_len, max_len)]
        logger.info(f"📊 文本长度过滤后: {len(df)} 行")
        
        # 删除重复项
        if preprocessing['remove_duplicates']:
            df = df.drop_duplicates(subset=[text_col])
            logger.info(f"📊 删除重复项后: {len(df)} 行")
        
        return df
    
    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射标签"""
        logger.info("🏷️ 开始标签映射...")
        
        label_col = self.data_config['field_mapping']['label_column']
        label_mapping = self.data_config['label_mapping']
        bert_labels = self.bert_config['labels']
        
        # 应用标签映射
        df['mapped_label'] = df[label_col].map(label_mapping)
        
        # 检查未映射的标签
        unmapped = df[df['mapped_label'].isna()]
        if len(unmapped) > 0:
            logger.warning(f"⚠️ 发现 {len(unmapped)} 个未映射的标签:")
            logger.warning(f"   {unmapped[label_col].unique()}")
            # 将未映射的标签设为OUT_OF_DOMAIN
            df['mapped_label'] = df['mapped_label'].fillna('OUT_OF_DOMAIN')
        
        # 添加数值标签
        df['label_id'] = df['mapped_label'].map(bert_labels)
        
        # 标签分布统计
        label_counts = df['mapped_label'].value_counts()
        logger.info("📊 标签分布:")
        for label, count in label_counts.items():
            logger.info(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割训练和验证数据"""
        logger.info("✂️ 分割训练和验证数据...")
        
        train_split = self.bert_config['data']['train_split']
        random_seed = self.bert_config['data']['random_seed']
        
        train_df, val_df = train_test_split(
            df,
            test_size=1-train_split,
            stratify=df['label_id'],
            random_state=random_seed
        )
        
        logger.info(f"📊 训练集: {len(train_df)} 样本")
        logger.info(f"📊 验证集: {len(val_df)} 样本")
        
        return train_df, val_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """保存处理后的数据"""
        logger.info("💾 保存处理后的数据...")
        
        text_col = self.data_config['field_mapping']['text_column']
        
        # 确保输出目录存在
        self.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备训练数据
        train_data = []
        for _, row in train_df.iterrows():
            train_data.append({
                "text": row[text_col],
                "label": row['mapped_label'],
                "label_id": row['label_id']
            })
        
        # 准备验证数据
        val_data = []
        for _, row in val_df.iterrows():
            val_data.append({
                "text": row[text_col],
                "label": row['mapped_label'], 
                "label_id": row['label_id']
            })
        
        # 保存为JSON
        with open(self.processed_train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(self.processed_val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 训练数据保存到: {self.processed_train_path}")
        logger.info(f"✅ 验证数据保存到: {self.processed_val_path}")
    
    def process(self):
        """执行完整的数据预处理流程"""
        logger.info("🚀 开始数据预处理流程...")
        
        try:
            # 1. 加载原始数据
            df = self.load_raw_data()
            
            # 2. 清洗数据
            df = self.clean_data(df)
            
            # 3. 映射标签
            df = self.map_labels(df)
            
            # 4. 分割数据
            train_df, val_df = self.split_data(df)
            
            # 5. 保存数据
            self.save_processed_data(train_df, val_df)
            
            logger.info("✅ 数据预处理完成！")
            
        except Exception as e:
            logger.error(f"❌ 数据预处理失败: {e}")
            raise

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python prepare_data.py <data_config.yaml> <bert_config.yaml>")
        sys.exit(1)
    
    data_config_path = sys.argv[1]
    bert_config_path = sys.argv[2]
    
    preprocessor = DataPreprocessor(data_config_path, bert_config_path)
    preprocessor.process()

if __name__ == "__main__":
    main()
