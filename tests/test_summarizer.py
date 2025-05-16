import pytest
from src.summarizer import TextSummarizer

@pytest.fixture
def summarizer():
    return TextSummarizer()

def test_summarizer_initialization(summarizer):
    """Test if the summarizer initializes correctly"""
    assert summarizer is not None
    assert summarizer.max_length == 130
    assert summarizer.min_length == 30

def test_summarize_single_text(summarizer):
    """Test single text summarization"""
    text = """
    Python is a high-level, general-purpose programming language. Its design philosophy emphasizes 
    code readability with the use of significant indentation. Python is dynamically typed and 
    garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, 
    and functional programming.
    """
    summary = summarizer.summarize(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(text)

def test_summarize_batch(summarizer):
    """Test batch summarization"""
    texts = [
        "First text about Python programming language.",
        "Second text about machine learning and artificial intelligence."
    ]
    summaries = summarizer.summarize_batch(texts)
    assert isinstance(summaries, list)
    assert len(summaries) == len(texts)
    for summary in summaries:
        assert isinstance(summary, str)
        assert len(summary) > 0

def test_custom_lengths(summarizer):
    """Test summarization with custom lengths"""
    text = "A short test text that needs to be summarized."
    summary = summarizer.summarize(text, max_length=50, min_length=10)
    assert isinstance(summary, str)
    assert len(summary) > 0 