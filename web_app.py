"""
Web-based UI for Code Generation and Summarization
Flask application with interactive interface
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src to path
sys.path.append('src')

from code_generator import CodeGenerator
from code_summarizer import CodeSummarizer

app = Flask(__name__)

# Initialize models with AI (better quality)
print("Initializing AI models...")
generator = CodeGenerator()
# Don't set model to None - use actual AI models for better code generation
summarizer = CodeSummarizer()
# Don't set model to None - use actual AI models for better summarization
print("Models ready!")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_code():
    """Generate code from description"""
    try:
        data = request.json
        description = data.get('description', '')
        language = data.get('language', 'python')
        
        if not description:
            return jsonify({'error': 'Please provide a description'}), 400
        
        # Generate code
        code = generator.generate_code(description, language)
        
        return jsonify({
            'success': True,
            'code': code,
            'language': language
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_variations', methods=['POST'])
def generate_variations():
    """Generate multiple code variations"""
    try:
        data = request.json
        description = data.get('description', '')
        language = data.get('language', 'python')
        num_variations = int(data.get('num_variations', 3))
        
        if not description:
            return jsonify({'error': 'Please provide a description'}), 400
        
        # Generate variations
        variations = generator.generate_multiple_variations(
            description, language, num_variations
        )
        
        return jsonify({
            'success': True,
            'variations': variations,
            'language': language
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_code():
    """Summarize code to natural language"""
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python')
        detailed = data.get('detailed', False)
        
        if not code:
            return jsonify({'error': 'Please provide code to summarize'}), 400
        
        if detailed:
            # Get detailed analysis
            details = summarizer.summarize_with_details(code, language)
            return jsonify({
                'success': True,
                'summary': details['overview'],
                'details': details
            })
        else:
            # Get basic summary
            summary = summarizer.summarize_code(code, language)
            return jsonify({
                'success': True,
                'summary': summary
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    """Analyze a file from the project"""
    try:
        data = request.json
        file_path = data.get('file_path', '')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Detect language from extension
        ext = os.path.splitext(file_path)[1]
        language_map = {'.py': 'python', '.java': 'java', '.js': 'javascript'}
        language = language_map.get(ext, 'python')
        
        # Get detailed analysis
        details = summarizer.summarize_with_details(code, language)
        
        return jsonify({
            'success': True,
            'file_path': file_path,
            'language': language,
            'summary': details['overview'],
            'details': details
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 CODE GENERATION & SUMMARIZATION WEB APP")
    print("="*80)
    print("\n📝 Features:")
    print("  • Generate code from English descriptions")
    print("  • Summarize code to natural language")
    print("  • Support for Python, Java, JavaScript")
    print("  • Generate multiple code variations")
    print("  • Detailed code analysis")
    print("\n🌐 Starting web server...")
    print("="*80)
    app.run(debug=True, host='127.0.0.1', port=5000)
