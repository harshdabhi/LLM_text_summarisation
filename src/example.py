from summarizer import TextSummarizer

def main():
    # Initialize the summarizer
    summarizer = TextSummarizer()
    
    # Example text to summarize
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
    intelligence displayed by animals including humans. AI research has been defined as the field 
    of study of intelligent agents, which refers to any system that perceives its environment and 
    takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" cognitive skills 
    that are associated with the human mind, such as "learning" and "problem-solving". This definition 
    has since been rejected by major AI researchers who now describe AI in terms of rationality and 
    acting rationally, which does not limit how intelligence can be articulated. AI applications include 
    advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon, and 
    Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), 
    generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the 
    highest level in strategic game systems (such as chess and Go).
    """
    
    # Generate summary
    summary = summarizer.summarize(text)
    
    print("Original text:")
    print(text)
    print("\nGenerated summary:")
    print(summary)

if __name__ == "__main__":
    main() 