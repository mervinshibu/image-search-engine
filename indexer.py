import math
from collections import defaultdict, Counter

class Indexer:
    def __init__(self, preprocessor):
        """
        Initialize the indexer.
        
        Args:
            preprocessor: The preprocessor object to use for text preprocessing
        """
        self.preprocessor = preprocessor
        self.inverted_index = defaultdict(dict)  # term -> {doc_id -> term_freq}
        self.document_lengths = {}  # doc_id -> document length (number of terms)
        self.avg_doc_length = 0
        self.doc_count = 0
        self.document_vectors = {}  # doc_id -> sparse vector of tf-idf weights
        self.document_norms = {}    # doc_id -> norm of document vector
        
    def add_document(self, doc_id, text):
        """
        Process a document and add it to the index.
        
        Args:
            doc_id: The document ID
            text: The document text
        """
        tokens = self.preprocessor.preprocess(text)
        
        # Count term frequencies in this document
        term_freqs = Counter(tokens)
        
        # Store document length (number of terms)
        self.document_lengths[doc_id] = len(tokens)
        
        # Update inverted index
        for term, freq in term_freqs.items():
            self.inverted_index[term][doc_id] = freq
        
        self.doc_count += 1
    
    def finalize_index(self):
        """
        Finalize the index by computing average document length
        and other required statistics.
        """
        if self.doc_count == 0:
            return
            
        # Calculate average document length
        total_length = sum(self.document_lengths.values())
        self.avg_doc_length = total_length / self.doc_count
        
        # Pre-compute document vectors for Vector Space Model
        self.build_document_vectors()
    
    def build_document_vectors(self):
        """
        Build TF-IDF document vectors for the Vector Space Model.
        """
        # For each term, calculate IDF
        idfs = {}
        for term, doc_freq_dict in self.inverted_index.items():
            df = len(doc_freq_dict)  # Document frequency
            idfs[term] = math.log(self.doc_count / df)
        
        # For each document, build its TF-IDF vector
        for doc_id in self.document_lengths:
            vector = {}
            
            for term, idf in idfs.items():
                if doc_id in self.inverted_index[term]:
                    tf = self.inverted_index[term][doc_id]
                    # Using log normalization for TF
                    if tf > 0:
                        normalized_tf = 1 + math.log(tf)
                        vector[term] = normalized_tf * idf
            
            self.document_vectors[doc_id] = vector
            
            # Calculate document vector norm
            self.document_norms[doc_id] = math.sqrt(sum(w**2 for w in vector.values()))
    
    def get_doc_count_for_term(self, term):
        """
        Get the number of documents containing the given term.
        
        Args:
            term: The term to look up
            
        Returns:
            int: Number of documents containing the term
        """
        return len(self.inverted_index.get(term, {}))
    
    def get_term_frequency(self, term, doc_id):
        """
        Get the frequency of a term in a specific document.
        
        Args:
            term: The term to look up
            doc_id: The document ID
            
        Returns:
            int: Frequency of the term in the document, or 0 if not found
        """
        return self.inverted_index.get(term, {}).get(doc_id, 0)
    
    def get_docs_for_term(self, term):
        """
        Get all documents containing the given term.
        
        Args:
            term: The term to look up
            
        Returns:
            dict: Dictionary mapping doc_id to term frequency
        """
        return self.inverted_index.get(term, {}) 