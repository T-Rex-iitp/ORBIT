"""
Local LLM Chat Web Application
Supports Ollama and custom model paths
"""
from flask import Flask, render_template, request, jsonify, Response
import requests
import json
import subprocess
import os
from models import model_manager
from rag import rag_manager

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models (Ollama + custom)"""
    ollama_url = request.args.get('ollama_url', model_manager.get_ollama_url())

    all_models = []

    # 1. Ollama 모델 가져오기
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            for m in data.get('models', []):
                all_models.append({
                    'name': m['name'],
                    'type': 'ollama',
                    'size': m.get('size', 0),
                    'description': f"Ollama: {m['name']}"
                })
    except:
        pass

    # 2. 커스텀 모델 추가
    for m in model_manager.get_custom_models():
        # 이미 ollama에 있는 모델인지 확인
        if not any(om['name'] == m['name'] for om in all_models):
            all_models.append({
                'name': m['name'],
                'type': 'custom',
                'path': m.get('path', ''),
                'description': m.get('description', '')
            })

    return jsonify({
        'success': True,
        'models': all_models,
        'default_model': model_manager.get_default_model(),
        'aliases': model_manager.config.get('model_aliases', {})
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get all configuration"""
    return jsonify({
        'success': True,
        'config': model_manager.get_all_config()
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration"""
    data = request.json
    model_manager.update_config(data)
    return jsonify({'success': True})

@app.route('/api/models/add', methods=['POST'])
def add_model():
    """Add a custom model"""
    data = request.json
    name = data.get('name', '')
    path = data.get('path', '')
    description = data.get('description', '')

    if not name:
        return jsonify({'success': False, 'error': 'Model name required'})

    result = model_manager.add_custom_model(name, path, description)
    return jsonify(result)

@app.route('/api/models/remove', methods=['POST'])
def remove_model():
    """Remove a custom model"""
    data = request.json
    name = data.get('name', '')

    if not name:
        return jsonify({'success': False, 'error': 'Model name required'})

    result = model_manager.remove_custom_model(name)
    return jsonify(result)

@app.route('/api/models/default', methods=['POST'])
def set_default_model():
    """Set default model"""
    data = request.json
    model_name = data.get('model', '')
    model_manager.set_default_model(model_name)
    return jsonify({'success': True})

@app.route('/api/models/alias', methods=['POST'])
def set_model_alias():
    """Set model alias"""
    data = request.json
    alias = data.get('alias', '')
    model_name = data.get('model', '')

    if not alias or not model_name:
        return jsonify({'success': False, 'error': 'Alias and model name required'})

    model_manager.set_model_alias(alias, model_name)
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the model (streaming)"""
    data = request.json
    model = data.get('model', '')
    messages = data.get('messages', [])
    ollama_url = data.get('ollama_url', model_manager.get_ollama_url())

    # 별칭 처리
    model = model_manager.get_model_alias(model)

    if not model:
        return jsonify({'success': False, 'error': 'No model selected'})

    # 모델별 설정 가져오기
    model_settings = model_manager.get_model_settings(model)

    def generate():
        try:
            request_body = {
                'model': model,
                'messages': messages,
                'stream': True
            }

            # 모델 설정 적용
            if model_settings:
                if 'temperature' in model_settings:
                    request_body['options'] = request_body.get('options', {})
                    request_body['options']['temperature'] = model_settings['temperature']

            response = requests.post(
                f"{ollama_url}/api/chat",
                json=request_body,
                stream=True,
                timeout=300
            )

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        if chunk.get('done', False):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Connection failed. Is Ollama running?'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/pull', methods=['POST'])
def pull_model():
    """Pull a model from Ollama"""
    data = request.json
    model_name = data.get('model', '')
    ollama_url = data.get('ollama_url', model_manager.get_ollama_url())

    if not model_name:
        return jsonify({'success': False, 'error': 'No model name provided'})

    def generate():
        try:
            response = requests.post(
                f"{ollama_url}/api/pull",
                json={'name': model_name, 'stream': True},
                stream=True,
                timeout=3600
            )

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/custom-model', methods=['POST'])
def load_custom_model():
    """Create Modelfile and load custom model"""
    data = request.json
    model_path = data.get('model_path', '')
    model_name = data.get('model_name', 'custom-model')
    description = data.get('description', '')
    ollama_url = data.get('ollama_url', model_manager.get_ollama_url())

    if not model_path:
        return jsonify({'success': False, 'error': 'No model path provided'})

    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': f'Model file not found: {model_path}'})

    # Create Modelfile
    modelfile_content = f'FROM {model_path}'
    modelfile_path = '/tmp/Modelfile'

    try:
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        # Create model using ollama create
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', modelfile_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            # 설정에도 추가
            model_manager.add_custom_model(model_name, model_path, description)
            return jsonify({'success': True, 'model_name': model_name})
        else:
            return jsonify({'success': False, 'error': result.stderr})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Model creation timed out'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# === RAG Endpoints ===

@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG statistics"""
    return jsonify({'success': True, 'stats': rag_manager.get_stats()})

@app.route('/api/rag/connect-gcs', methods=['POST'])
def connect_gcs():
    """Connect to Google Cloud Storage"""
    data = request.json
    credentials_path = data.get('credentials_path')
    bucket_name = data.get('bucket_name')

    if not bucket_name:
        return jsonify({'success': False, 'error': 'Bucket name required'})

    result = rag_manager.connect_gcs(credentials_path, bucket_name)
    return jsonify(result)

@app.route('/api/rag/gcs-files', methods=['GET'])
def list_gcs_files():
    """List files in GCS bucket"""
    prefix = request.args.get('prefix', '')
    extensions = request.args.get('extensions', '.txt,.pdf,.md,.json').split(',')
    result = rag_manager.list_gcs_files(prefix=prefix, extensions=extensions)
    return jsonify(result)

@app.route('/api/rag/index-file', methods=['POST'])
def index_gcs_file():
    """Index a single file from GCS"""
    data = request.json
    blob_name = data.get('blob_name', '')

    if not blob_name:
        return jsonify({'success': False, 'error': 'Blob name required'})

    result = rag_manager.index_gcs_file(blob_name)
    return jsonify(result)

@app.route('/api/rag/index-folder', methods=['POST'])
def index_gcs_folder():
    """Index all files in a GCS folder"""
    data = request.json
    prefix = data.get('prefix', '')
    extensions = data.get('extensions', ['.txt', '.pdf', '.md', '.json'])

    result = rag_manager.index_gcs_folder(prefix=prefix, extensions=extensions)
    return jsonify(result)

@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    """Search indexed documents"""
    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 5)

    if not query:
        return jsonify({'success': False, 'error': 'Query required'})

    results = rag_manager.search(query, n_results=n_results)
    return jsonify({'success': True, 'results': results})

@app.route('/api/rag/delete', methods=['POST'])
def rag_delete():
    """Delete a document from index"""
    data = request.json
    doc_id = data.get('doc_id', '')

    if not doc_id:
        return jsonify({'success': False, 'error': 'Document ID required'})

    result = rag_manager.delete_document(doc_id)
    return jsonify(result)

# === BigQuery Endpoints ===

@app.route('/api/rag/connect-bq', methods=['POST'])
def connect_bigquery():
    """Connect to BigQuery"""
    data = request.json
    project_id = data.get('project_id', '')
    dataset = data.get('dataset', '')
    credentials_path = data.get('credentials_path')

    if not project_id:
        return jsonify({'success': False, 'error': 'Project ID required'})

    result = rag_manager.connect_bigquery(project_id, dataset, credentials_path)
    return jsonify(result)

@app.route('/api/rag/bq-tables', methods=['GET'])
def list_bq_tables():
    """List BigQuery tables"""
    result = rag_manager.list_bq_tables()
    return jsonify(result)

@app.route('/api/rag/bq-schema', methods=['GET'])
def get_bq_schema():
    """Get BigQuery table schema"""
    table_name = request.args.get('table', '')
    if not table_name:
        return jsonify({'success': False, 'error': 'Table name required'})
    result = rag_manager.get_bq_schema(table_name)
    return jsonify(result)

@app.route('/api/rag/bq-query', methods=['POST'])
def query_bigquery():
    """Execute BigQuery SQL"""
    data = request.json
    sql = data.get('sql', '')
    max_results = data.get('max_results', 100)

    if not sql:
        return jsonify({'success': False, 'error': 'SQL query required'})

    result = rag_manager.query_bigquery(sql, max_results)
    return jsonify(result)

@app.route('/api/rag/index-bq', methods=['POST'])
def index_bq_table():
    """Index BigQuery table for RAG"""
    data = request.json
    table_name = data.get('table', '')
    text_columns = data.get('columns', [])
    id_column = data.get('id_column')
    limit = data.get('limit', 1000)

    if not table_name or not text_columns:
        return jsonify({'success': False, 'error': 'Table and columns required'})

    result = rag_manager.index_bq_table(table_name, text_columns, id_column, limit)
    return jsonify(result)

@app.route('/api/chat-rag', methods=['POST'])
def chat_with_rag():
    """Chat with RAG context (streaming)"""
    data = request.json
    model = data.get('model', '')
    messages = data.get('messages', [])
    use_rag = data.get('use_rag', True)
    n_context = data.get('n_context', 3)
    ollama_url = data.get('ollama_url', model_manager.get_ollama_url())

    model = model_manager.get_model_alias(model)

    if not model:
        return jsonify({'success': False, 'error': 'No model selected'})

    # Get the last user message for RAG
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_user_msg = msg.get('content', '')
            break

    # Apply RAG if enabled
    rag_context = None
    if use_rag and last_user_msg:
        rag_result = rag_manager.rag_query(last_user_msg, n_context=n_context)
        if rag_result.get('has_context'):
            rag_context = rag_result
            # Replace last user message with augmented prompt
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    messages[i]['content'] = rag_result['prompt']
                    break

    model_settings = model_manager.get_model_settings(model)

    def generate():
        # Send RAG context info first
        if rag_context:
            sources = [c['metadata'].get('filename', 'unknown') for c in rag_context['context']]
            yield f"data: {json.dumps({'rag_sources': sources})}\n\n"

        try:
            request_body = {
                'model': model,
                'messages': messages,
                'stream': True
            }

            if model_settings:
                if 'temperature' in model_settings:
                    request_body['options'] = request_body.get('options', {})
                    request_body['options']['temperature'] = model_settings['temperature']

            response = requests.post(
                f"{ollama_url}/api/chat",
                json=request_body,
                stream=True,
                timeout=300
            )

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                        if chunk.get('done', False):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Connection failed. Is Ollama running?'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    print("=" * 50)
    print("  Local LLM Chat Web Application + RAG")
    print("=" * 50)
    print(f"\n  Open http://localhost:5000 in your browser")
    print(f"\n  Default model: {model_manager.get_default_model() or 'Not set'}")
    print(f"  Custom models: {len(model_manager.get_custom_models())}")
    print(f"  RAG documents: {rag_manager.get_stats().get('document_count', 0)}")
    print(f"\n  Make sure Ollama is running: ollama serve")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
