#!/usr/bin/env python3
"""
调试spaCy实体识别结果
查看多个实体的识别情况
"""
import spacy

def debug_spacy_entities():
    """调试spaCy实体识别"""
    print("🔍 调试spaCy实体识别...")
    
    try:
        # 加载spaCy模型
        nlp = spacy.load("en_core_web_sm")
        
        # 测试查询
        test_queries = [
            "How old is Kobe Bryant?",
            "How old is Kobe Bryant and Yao Ming?",
            "Compare Kobe Bryant and Michael Jordan",
            "Tell me about Lakers and Bulls teams"
        ]
        
        for query in test_queries:
            print(f"\n📝 查询: '{query}'")
            print("-" * 50)
            
            doc = nlp(query)
            
            print("=== Tokens ===")
            for i, token in enumerate(doc):
                print(f"{i:2d}. '{token.text:12s}' (pos: {token.pos_:8s}, ent_type: '{token.ent_type_:8s}')")
            
            print("\n=== Entities ===")
            if doc.ents:
                for ent in doc.ents:
                    print(f"'{ent.text}' ({ent.label_}) - span({ent.start}, {ent.end})")
            else:
                print("❌ 没有识别到任何实体")
            
            print("\n" + "="*60)
    
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_spacy_entities()
