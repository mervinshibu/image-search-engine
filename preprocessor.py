import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self):
        """Initialize the preprocessor with stopwords and stemmer."""
        # Download required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess(self, text):
        """
        Preprocess the text by:
        1. Converting to lowercase
        2. Tokenizing
        3. Removing stopwords
        4. Stemming
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        if not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and replace with space
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                processed_tokens.append(self.stemmer.stem(token))
                
        return processed_tokens 