#!/usr/bin/env python3
"""
模型导出脚本
将训练好的模型导出为生产可用格式
"""
import os
import sys
import shutil
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self):
        self.models_dir = Path("../../models/intent_classifier")
        self.training_output = Path("../../models/intent_classifier/v1.0/final")
        
    def export_to_production(self, version: str = "v1.0"):
        """导出模型到生产环境"""
        logger.info(f"📦 导出模型到生产环境 (版本: {version})")
        
        # 检查训练输出是否存在
        if not self.training_output.exists():
            logger.error(f"❌ 训练输出不存在: {self.training_output}")
            logger.error("请先运行训练脚本")
            return False
        
        try:
            # 创建版本目录
            version_dir = self.models_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制模型文件
            logger.info("📁 复制模型文件...")
            for file_name in ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]:
                src_file = self.training_output / file_name
                if src_file.exists():
                    dst_file = version_dir / file_name
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"   ✅ {file_name}")
                else:
                    logger.warning(f"   ⚠️ 文件不存在: {file_name}")
            
            # 创建current软链接
            current_dir = self.models_dir / "current"
            if current_dir.exists():
                if current_dir.is_symlink():
                    current_dir.unlink()
                else:
                    shutil.rmtree(current_dir)
            
            # 创建相对路径软链接
            current_dir.symlink_to(version, target_is_directory=True)
            
            logger.info(f"✅ 模型导出完成!")
            logger.info(f"📁 版本目录: {version_dir}")
            logger.info(f"🔗 当前链接: {current_dir} -> {version}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型导出失败: {e}")
            return False
    
    def create_model_info(self, version: str):
        """创建模型信息文件"""
        version_dir = self.models_dir / version
        info_file = version_dir / "model_info.txt"
        
        info_content = f"""BERT意图分类模型信息
===================

版本: {version}
基础模型: distilbert-base-uncased
任务: 篮球问答意图分类
类别数: 5

支持的意图类别:
- ATTRIBUTE_QUERY: 属性查询
- SIMPLE_RELATION_QUERY: 简单关系查询  
- COMPARATIVE_QUERY: 比较查询
- DOMAIN_CHITCHAT: 篮球闲聊
- OUT_OF_DOMAIN: 领域外查询

使用方法:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("path/to/model")
model = AutoModelForSequenceClassification.from_pretrained("path/to/model")
```

训练时间: {import datetime; datetime.datetime.now().isoformat()}
"""
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        logger.info(f"📝 模型信息保存到: {info_file}")

def main():
    """主函数"""
    version = sys.argv[1] if len(sys.argv) > 1 else "v1.0"
    
    exporter = ModelExporter()
    
    if exporter.export_to_production(version):
        exporter.create_model_info(version)
        print(f"\n🎉 模型导出成功!")
        print(f"📁 版本: {version}")
        print(f"🔗 当前版本链接已更新")
        print(f"💡 你现在可以在业务代码中使用该模型")
    else:
        print("❌ 模型导出失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
