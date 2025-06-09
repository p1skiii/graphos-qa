"""
API路由定义
"""
from flask import Blueprint, request, jsonify, render_template_string
from app.rag import rag_system

# 创建蓝图
api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def home():
    """首页"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>篮球知识问答系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
            .sources { margin-top: 15px; }
            .source-item { background-color: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏀 篮球知识问答系统</h1>
            <div class="form-group">
                <label for="question">请输入您的问题:</label>
                <input type="text" id="question" placeholder="例如：LeBron James多少岁？" />
            </div>
            <button onclick="askQuestion()">提问</button>
            <div id="result"></div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question.trim()) {
                    alert('请输入问题');
                    return;
                }
                
                document.getElementById('result').innerHTML = '<p>🤔 正在思考...</p>';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    let html = '<div class="result">';
                    html += '<h3>📝 回答:</h3>';
                    html += '<p>' + data.answer + '</p>';
                    html += '<p><strong>置信度:</strong> ' + (data.confidence * 100).toFixed(1) + '%</p>';
                    
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources">';
                        html += '<h4>📚 相关信息来源:</h4>';
                        data.sources.forEach(source => {
                            html += '<div class="source-item">';
                            html += '<strong>类型:</strong> ' + source.type + '<br>';
                            html += '<strong>内容:</strong> ' + source.text + '<br>';
                            html += '<strong>相似度:</strong> ' + (source.similarity * 100).toFixed(1) + '%';
                            html += '</div>';
                        });
                        html += '</div>';
                    }
                    html += '</div>';
                    
                    document.getElementById('result').innerHTML = html;
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div class="result">❌ 请求失败: ' + error.message + '</div>';
                }
            }
            
            // 支持回车键提交
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """)

@api_bp.route('/ask', methods=['POST'])
def ask_question():
    """处理问答请求"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
        
        # 使用RAG系统回答问题
        result = rag_system.answer_question(question)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500

@api_bp.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_system.embedding_model is not None
    })

@api_bp.route('/knowledge-base')
def get_knowledge_base():
    """获取知识库信息"""
    return jsonify({
        'total_items': len(rag_system.knowledge_base),
        'sample_items': rag_system.knowledge_base[:5] if rag_system.knowledge_base else []
    })
