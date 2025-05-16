# LLM Text Summarization Project

This project provides a simple and efficient way to generate summaries of text using state-of-the-art language models. It uses the Hugging Face Transformers library and pre-trained models to generate high-quality summaries.

## Features

- Easy-to-use API for text summarization
- Support for both single text and batch summarization
- Configurable summary length
- GPU acceleration support
- Error handling and logging
- Beautiful and responsive web interface
- Dark mode support

## Video Demonstration

[![Watch the demo](https://img.youtube.com/vi/UC2pzhdhVRU/0.jpg)](https://youtu.be/UC2pzhdhVRU)

Installation

1. Clone the repository:

```bash
git clone https://github.com/harshdabhi/LLM_text_summarisation.git
cd llm_summarisation
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use the summarizer:

```python
from src.summarizer import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Generate a summary
text = "Your long text here..."
summary = summarizer.summarize(text)
print(summary)
```

For batch processing:

```python
texts = ["First long text...", "Second long text..."]
summaries = summarizer.summarize_batch(texts)
```

### Web Interface

To use the web interface:

1. Start the Flask server:

```bash
python src/app.py
```

2. Open your browser and navigate to `http://localhost:5000`
3. Enter your text and adjust the summary length parameters
4. Click "Summarize" to generate the summary

## Configuration

The `TextSummarizer` class accepts the following parameters:

- `model_name`: The name of the pre-trained model to use (default: "facebook/bart-large-cnn")
- `device`: The device to run the model on ('cuda' or 'cpu')
- `max_length`: Maximum length of the generated summary (default: 130)
- `min_length`: Minimum length of the generated summary (default: 30)

## Project Structure

```
llm_summarisation/
├── src/
│   ├── summarizer.py    # Main summarization module
│   ├── app.py          # Flask web application
│   ├── static/         # Static files (CSS, JS)
│   │   └── css/
│   │       └── style.css
│   └── templates/      # HTML templates
│       └── index.html
├── tests/              # Test files
├── data/              # Data directory
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
