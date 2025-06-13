#!/usr/bin/env python
"""
简单的NLP模块测试
验证基础NLP功能是否正常工作
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

def test_nlp_components():
    """测试NLP组件"""
    print("🧠 测试NLP模块...")
    
    try:
        # 导入模块
        from app.core.schemas import QueryContextFactory
        from app.services.nlp import (
            LanguageDetector, 
            Tokenizer, 
            EntityExtractor, 
            IntentClassifier,
            NLPPipeline
        )
        print("✅ NLP模块导入成功")
        
        # 创建各个组件
        print("\n📝 初始化NLP组件...")
        language_detector = LanguageDetector()
        tokenizer = Tokenizer()
        entity_extractor = EntityExtractor() 
        intent_classifier = IntentClassifier()
        
        # 初始化组件
        components = [
            ("语言检测器", language_detector),
            ("分词器", tokenizer),
            ("实体提取器", entity_extractor),
            ("意图分类器", intent_classifier)
        ]
        
        for name, component in components:
            print(f"初始化 {name}...")
            if component.initialize():
                print(f"✅ {name} 初始化成功")
            else:
                print(f"❌ {name} 初始化失败")
                return
        
        # 创建NLP管道
        print("\n🔄 创建NLP处理管道...")
        nlp_pipeline = NLPPipeline(
            language_detector=language_detector,
            tokenizer=tokenizer,
            entity_extractor=entity_extractor,
            intent_classifier=intent_classifier
        )
        
        if nlp_pipeline.initialize():
            print("✅ NLP管道初始化成功")
        else:
            print("❌ NLP管道初始化失败")
            return
        
        # 测试查询
        test_query = "Who was in the Rockets, Kobe or Yao Ming? ?"
        print(f"\n🔍 测试查询: '{test_query}'")
        print("-" * 50)
        
        # 创建QueryContext
        context = QueryContextFactory.create(test_query)
        print(f"📄 原始查询: {context.original_query}")
        
        # 逐步处理
        print("\n🌐 步骤1: 语言检测...")
        context = language_detector.process(context)
        if context.language_info:
            print(f"   检测语言: {context.language_info.original_language}")
            print(f"   置信度: {context.language_info.detected_confidence:.2f}")
        
        print("\n🔤 步骤2: 分词...")
        context = tokenizer.process(context)
        if hasattr(context, 'tokens') and context.tokens:
            print(f"   分词结果: {len(context.tokens)} 个token")
            
            # 🆕 显示所有重要token（前5个 + 有实体类型的）
            shown_indices = set()
            
            # 显示前5个
            for i, token in enumerate(context.tokens[:5]):
                print(f"   {i+1}. '{token.text}' ({token.pos}) [ent_type: '{token.ent_type}']")
                shown_indices.add(i)
            
            # 显示剩余的有实体类型的token
            for i, token in enumerate(context.tokens[5:], 5):
                if token.ent_type or token.pos == "PROPN":
                    print(f"   {i+1}. '{token.text}' ({token.pos}) [ent_type: '{token.ent_type}']")
        
        print("\n👤 步骤3: 实体提取...")
        context = entity_extractor.process(context)
        if context.entity_info:
            print(f"   球员: {context.entity_info.players}")
            print(f"   队伍: {context.entity_info.teams}")
            print(f"   所有目标实体: {context.entity_info.target_entities}")
            print(f"   向后兼容目标: {context.entity_info.target_entity}")
        
        print("\n🎯 步骤4: 意图分类...")
        context = intent_classifier.process(context)
        if context.intent_info:
            print(f"   意图: {context.intent_info.intent}")
            print(f"   置信度: {context.intent_info.confidence:.2f}")
            print(f"   属性类型: {context.intent_info.attribute_type}")
        
        print("\n🔄 测试完整管道...")
        # 重新创建context测试完整管道
        context2 = QueryContextFactory.create(test_query)
        result = nlp_pipeline.process(context2)
        
        print(f"✅ 管道处理完成!")
        print(f"   最终意图: {result.intent_info.intent if result.intent_info else 'None'}")
        print(f"   目标实体: {result.entity_info.target_entity if result.entity_info else 'None'}")
        
        print("\n🎉 NLP模块测试成功!")
        
        # 🆕 清理资源
        print("\n🔄 清理资源...")
        try:
            entity_extractor.close()  # 关闭NebulaGraph连接
            print("✅ 资源清理完成")
        except Exception as e:
            print(f"⚠️ 资源清理警告: {e}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nlp_components()
