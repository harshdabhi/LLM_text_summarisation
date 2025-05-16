from flask import Flask, render_template, request, jsonify
from summarizer import TextSummarizer
import logging

app = Flask(__name__)
summarizer = TextSummarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = int(data.get('max_length', 130))
        min_length = int(data.get('min_length', 30))
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        summary = summarizer.summarize(
            text,
            max_length=max_length,
            min_length=min_length
        )
        
        return jsonify({
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        })
        
    except Exception as e:
        logging.error(f"Error during summarization: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 