"""
Flask应用主文件
"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from app.database.nebula_connection import nebula_conn
import os

app = Flask(__name__)
CORS(app)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NebulaGraph Flask 应用</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .query-section { margin: 20px 0; }
        .query-input { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; }
        .btn { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background-color: #0056b3; }
        .result { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 15px; margin: 10px 0; }
        .table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 NebulaGraph Flask 应用</h1>
        
        <div id="status" class="status">正在检查连接状态...</div>
        
        <div class="query-section">
            <h3>执行nGQL查询</h3>
            <textarea id="queryInput" class="query-input" placeholder="输入您的nGQL查询语句，例如：SHOW TAGS"></textarea>
            <br><br>
            <button class="btn" onclick="executeQuery()">执行查询</button>
            <button class="btn" onclick="loadSampleQueries()" style="margin-left: 10px; background-color: #28a745;">加载示例查询</button>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h4>查询结果:</h4>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        // 检查连接状态
        fetch('/api/test')
            .then(response => response.json())
            .then(data => {
                const statusDiv = document.getElementById('status');
                if (data.success) {
                    statusDiv.className = 'status success';
                    statusDiv.innerHTML = '✅ 成功连接到NebulaGraph (Space: basketballplayer)';
                } else {
                    statusDiv.className = 'status error';
                    statusDiv.innerHTML = '❌ 连接失败: ' + data.error;
                }
            });

        function executeQuery() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('请输入查询语句');
                return;
            }

            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                resultDiv.style.display = 'block';
                
                if (data.success) {
                    if (data.data.rows && data.data.rows.length > 0) {
                        let html = '<table class="table"><thead><tr>';
                        data.data.column_names.forEach(col => {
                            html += `<th>${col}</th>`;
                        });
                        html += '</tr></thead><tbody>';
                        
                        data.data.rows.forEach(row => {
                            html += '<tr>';
                            row.forEach(cell => {
                                html += `<td>${JSON.stringify(cell)}</td>`;
                            });
                            html += '</tr>';
                        });
                        html += '</tbody></table>';
                        resultContent.innerHTML = html;
                    } else {
                        resultContent.innerHTML = '<p>查询执行成功，但没有返回数据。</p>';
                    }
                } else {
                    resultContent.innerHTML = `<div class="error">查询错误: ${data.error}</div>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('resultContent').innerHTML = `<div class="error">请求失败: ${error}</div>`;
            });
        }

        function loadSampleQueries() {
            const samples = [
                'SHOW TAGS',
                'SHOW EDGES', 
                'MATCH (v:player) RETURN v.player.name LIMIT 10',
                'MATCH (v:team) RETURN v.team.name LIMIT 5'
            ];
            
            let sampleText = '// 以下是一些示例查询，您可以选择其中一个测试：\\n\\n';
            samples.forEach((query, index) => {
                sampleText += `// 示例 ${index + 1}: ${query}\\n`;
            });
            sampleText += '\\n// 请删除注释并选择一个查询执行\\n';
            sampleText += samples[0]; // 默认选择第一个
            
            document.getElementById('queryInput').value = sampleText;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/test', methods=['GET'])
def test_connection():
    """测试NebulaGraph连接"""
    try:
        if nebula_conn.connect():
            success = nebula_conn.test_connection()
            return jsonify({
                'success': success,
                'message': '连接测试完成' if success else '连接测试失败'
            })
        else:
            return jsonify({
                'success': False,
                'error': '无法建立连接'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/query', methods=['POST'])
def execute_query():
    """执行nGQL查询"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查询语句不能为空'
            })
        
        # 确保连接存在
        if not nebula_conn.session:
            if not nebula_conn.connect():
                return jsonify({
                    'success': False,
                    'error': '无法连接到NebulaGraph'
                })
        
        # 执行查询
        result = nebula_conn.execute_query(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/info', methods=['GET'])
def get_graph_info():
    """获取图数据库信息"""
    try:
        if not nebula_conn.session:
            if not nebula_conn.connect():
                return jsonify({
                    'success': False,
                    'error': '无法连接到NebulaGraph'
                })
        
        # 获取标签信息
        tags_result = nebula_conn.execute_query("SHOW TAGS")
        edges_result = nebula_conn.execute_query("SHOW EDGES")
        spaces_result = nebula_conn.execute_query("SHOW SPACES")
        
        return jsonify({
            'success': True,
            'data': {
                'tags': tags_result['rows'] if tags_result['success'] else [],
                'edges': edges_result['rows'] if edges_result['success'] else [],
                'spaces': spaces_result['rows'] if spaces_result['success'] else []
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    try:
        # 启动时测试连接
        print("正在启动Flask应用...")
        print("尝试连接到NebulaGraph...")
        
        if nebula_conn.connect():
            print("✅ NebulaGraph连接成功!")
            nebula_conn.test_connection()
        else:
            print("❌ NebulaGraph连接失败，请检查配置")
            
        # 启动Flask应用
        port = int(os.getenv('PORT', 5000))
        app.run(debug=True, host='0.0.0.0', port=port)
        
    except KeyboardInterrupt:
        print("\\n正在关闭应用...")
    finally:
        nebula_conn.close()
