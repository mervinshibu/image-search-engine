import math
from collections import defaultdict

class SearchEngine:
    def __init__(self, indexer, preprocessor):
        """
        Initialize the search engine.
        
        Args:
            indexer: The indexer object with document data
            preprocessor: The preprocessor for query processing
        """
        self.indexer = indexer
        self.preprocessor = preprocessor
        
    def search_vsm(self, query, top_k=100):
        """
        Search using Vector Space Model (TF-IDF).
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of (doc_id, score) tuples sorted by decreasing score
        """
        # Preprocess the query
        query_terms = self.preprocessor.preprocess(query)
        
        # Build query vector using same weighting as documents
        query_vector = defaultdict(float)
        for term in query_terms:
            if term in self.indexer.inverted_index:
                # Term frequency in query
                tf = query_terms.count(term)
                # Log normalization for TF
                normalized_tf = 1 + math.log(tf)
                # IDF computation same as document indexing
                df = self.indexer.get_doc_count_for_term(term)
                idf = math.log(self.indexer.doc_count / df)
                # TF-IDF weight for query term
                query_vector[term] = normalized_tf * idf
        
        # Calculate query vector norm
        query_norm = math.sqrt(sum(w**2 for w in query_vector.values()))
        
        # Calculate cosine similarity for each document
        scores = {}
        for doc_id, doc_vector in self.indexer.document_vectors.items():
            dot_product = 0.0
            for term, weight in query_vector.items():
                if term in doc_vector:
                    dot_product += weight * doc_vector[term]
            
            # Avoid division by zero
            if query_norm == 0 or self.indexer.document_norms[doc_id] == 0:
                scores[doc_id] = 0
            else:
                # Cosine similarity
                scores[doc_id] = dot_product / (query_norm * self.indexer.document_norms[doc_id])
        
        # Sort by decreasing score and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def search_bm25(self, query, top_k=100, k1=1.2, b=0.75):
        """
        Search using BM25 ranking algorithm.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            k1 (float): Term frequency saturation parameter
            b (float): Document length normalization parameter
            
        Returns:
            list: List of (doc_id, score) tuples sorted by decreasing score
        """
        # Preprocess the query
        query_terms = self.preprocessor.preprocess(query)
        
        # Calculate scores using BM25 formula
        scores = defaultdict(float)
        
        for term in query_terms:
            if term not in self.indexer.inverted_index:
                continue
                
            # Calculate IDF component of BM25
            n = self.indexer.doc_count
            df = self.indexer.get_doc_count_for_term(term)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            
            # For each document containing this term
            for doc_id, tf in self.indexer.get_docs_for_term(term).items():
                doc_length = self.indexer.document_lengths[doc_id]
                avg_doc_length = self.indexer.avg_doc_length
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort by decreasing score and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def search_lm_dirichlet(self, query, top_k=100, mu=2000):
        """
        Search using Language Model with Dirichlet smoothing.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            mu (float): Dirichlet smoothing parameter
            
        Returns:
            list: List of (doc_id, score) tuples sorted by decreasing score
        """
        # Preprocess the query
        query_terms = self.preprocessor.preprocess(query)
        
        # Compute collection statistics for smoothing
        total_terms = sum(self.indexer.document_lengths.values())
        
        # For each term, compute collection probability
        collection_prob = {}
        for term in set(query_terms):
            if term in self.indexer.inverted_index:
                # Count total occurrences of term in collection
                term_count = sum(self.indexer.inverted_index[term].values())
                collection_prob[term] = term_count / total_terms
            else:
                collection_prob[term] = 1 / total_terms  # Smoothing for unseen terms
        
        # Calculate scores using Language Model with Dirichlet smoothing
        scores = {}
        
        for doc_id, doc_length in self.indexer.document_lengths.items():
            score = 0.0
            
            for term in query_terms:
                # Term frequency in document
                tf = self.indexer.get_term_frequency(term, doc_id)
                
                # Calculate p(term|document) with Dirichlet smoothing
                p_term_given_doc = (tf + mu * collection_prob.get(term, 0)) / (doc_length + mu)
                
                # Avoid log(0)
                if p_term_given_doc > 0:
                    # Log probability (using log sum instead of product for numerical stability)
                    score += math.log(p_term_given_doc)
                else:
                    score -= 100  # Penalize heavily for missing terms
            
            scores[doc_id] = score
        
        # Sort by decreasing score and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k] 