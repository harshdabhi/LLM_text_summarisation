from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_length: int = 130,
        min_length: int = 30,
    ):
        """
        Initialize the text summarizer with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum length of the generated summary
            min_length: Minimum length of the generated summary
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_length = min_length
        
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Initialize the summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """
        Generate a summary for the given text.
        
        Args:
            text: Input text to summarize
            max_length: Optional override for max_length
            min_length: Optional override for min_length
            
        Returns:
            Generated summary as a string
        """
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length
        
        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]["summary_text"]
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise
            
    def summarize_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts to summarize
            max_length: Optional override for max_length
            min_length: Optional override for min_length
            
        Returns:
            List of generated summaries
        """
        max_length = max_length or self.max_length
        min_length = min_length or self.min_length
        
        try:
            summaries = self.summarizer(
                texts,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return [summary["summary_text"] for summary in summaries]
        except Exception as e:
            logger.error(f"Error during batch summarization: {str(e)}")
            raise 