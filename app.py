# app.py

import os
import json
import pickle
import time
import sys
from flask import Flask, render_template, request, url_for, redirect

# --- Assuming your A1 files are accessible ---
try:
    from preprocessor import Preprocessor
    from indexer import Indexer
    from search import SearchEngine
except ImportError as e:
    print(f"Error importing assignment 1 modules: {e}")
    print("Make sure preprocessor.py, indexer.py, and search.py are in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# --- Configuration ---
INDEX_DIR = 'image_index_data' # Directory where index files are stored
INDEX_COMPONENTS_FILE = os.path.join(INDEX_DIR, 'index_components.pkl')
DOC_ID_MAP_FILE = os.path.join(INDEX_DIR, 'doc_id_map.json')

# --- Initialize Flask App ---
app = Flask(__name__) # Standard Flask app initialization

# --- Global Variables to hold loaded index data ---
search_engine = None
doc_id_map = None

# --- Load Index Data ONCE at Startup ---
def load_index_data(index_file, map_file):
    """Loads the index data structures and ID map."""
    print(f"Loading index components from {index_file}...")
    try:
        with open(index_file, 'rb') as f:
            loaded_index_data = pickle.load(f)
        print("Index components loaded.")
    except FileNotFoundError:
        print(f"Error: Index components file not found at {index_file}")
        return None, None
    except Exception as e:
        print(f"Error loading index components: {e}")
        return None, None

    print(f"Loading document ID mapping from {map_file}...")
    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            loaded_doc_id_map = json.load(f)
            # Convert JSON string keys back to integers
            loaded_doc_id_map = {int(k): v for k, v in loaded_doc_id_map.items()}
        print("Document ID mapping loaded.")
    except FileNotFoundError:
        print(f"Error: Document ID mapping file not found at {map_file}")
        return None, None
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return None, None

    # Reconstruct the Indexer object and create SearchEngine
    try:
        print("Reconstructing indexer and initializing SearchEngine...")
        preprocessor = Preprocessor() # Preprocessor is needed for SearchEngine
        reconstructed_indexer = Indexer(preprocessor)

        # Populate the reconstructed indexer with loaded data
        reconstructed_indexer.inverted_index = loaded_index_data['inverted_index']
        reconstructed_indexer.document_lengths = loaded_index_data['document_lengths']
        reconstructed_indexer.doc_count = loaded_index_data['doc_count']
        reconstructed_indexer.avg_doc_length = loaded_index_data['avg_doc_length']
        reconstructed_indexer.document_vectors = loaded_index_data['document_vectors']
        reconstructed_indexer.document_norms = loaded_index_data['document_norms']

        # Create the search engine instance
        loaded_search_engine = SearchEngine(reconstructed_indexer, preprocessor)
        print("SearchEngine initialized successfully.")
        return loaded_search_engine, loaded_doc_id_map
    except Exception as e:
        print(f"Error reconstructing indexer object or initializing SearchEngine: {e}")
        return None, None

# --- Load the data when the Flask app starts ---
print("="*30)
print("Attempting to load search index...")
search_engine, doc_id_map = load_index_data(INDEX_COMPONENTS_FILE, DOC_ID_MAP_FILE)
if not search_engine or not doc_id_map:
    print("FATAL: Could not load index data. Please ensure index files exist in:")
    print(f"  {INDEX_COMPONENTS_FILE}")
    print(f"  {DOC_ID_MAP_FILE}")
    print("You may need to run the index building script first.")
    sys.exit(1) # Exit if index cannot be loaded
print("Index loaded successfully.")
print("="*30)


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main search page without any results initially."""
    # Redirect to the search page to keep URL consistent
    return redirect(url_for('search'))

@app.route('/search')
def search():
    """
    Handles displaying the search form and processing search requests.
    Search parameters (query, model) are expected in the URL (GET request).
    """
    query = request.args.get('query', '') # Get query from URL parameter, default to empty
    model = request.args.get('model', 'bm25') # Get model, default to bm25
    top_k = 50 # Number of results to fetch

    results = []
    formatted_results = []
    model_display_name = {
        'vsm': 'TF-IDF Portal Gun', # Matches new HTML labels
        'bm25': 'BM25 Meeseeks Box', # Matches new HTML labels
        'lm_dirichlet': 'Language Model Microverse' # Matches new HTML labels
    }.get(model, 'Unknown Dimension') # Default themed name

    search_performed = bool(query) # Check if a query was actually submitted

    if search_performed:
        print(f"Received search request: query='{query}', model='{model}'")
        start_time = time.time()
        try:
            if model == 'vsm':
                results = search_engine.search_vsm(query, top_k=top_k)
            elif model == 'bm25':
                results = search_engine.search_bm25(query, top_k=top_k)
            elif model == 'lm_dirichlet':
                results = search_engine.search_lm_dirichlet(query, top_k=top_k)
            else:
                print(f"Warning: Unknown model '{model}' requested.")
                # Optionally, add an error message to pass to the template

            end_time = time.time()
            print(f"Search completed in {end_time - start_time:.4f} seconds, found {len(results)} raw results.")

            # Format results using the doc_id_map
            for rank, (doc_id, score) in enumerate(results, 1):
                metadata = doc_id_map.get(doc_id) # Use .get for safety
                if metadata:
                    formatted_results.append({
                        'rank': rank,
                        'score': score,
                        'image_url': metadata['image_url'],
                        'source_page': metadata['source_page'],
                        'alt_text': metadata.get('alt_text', f'Search result for {query}') # Add default alt text
                    })
                else:
                    print(f"Warning: Could not find metadata for doc_id {doc_id}")

        except Exception as e:
            print(f"Error during search execution for query '{query}' model '{model}': {e}")
            # Optionally, pass an error message to the template

    # Render the same HTML template, passing necessary data
    return render_template(
        'rick-morty-search-ui.html', # Use the new HTML file name
        query=query,
        selected_model=model,
        results=formatted_results,
        model_display_name=model_display_name,
        search_performed=search_performed
    )


# --- Run the Flask App ---
if __name__ == '__main__':
    # debug=True automatically restarts the server when you save changes
    # Set host='0.0.0.0' to make it accessible on your network (optional)
    app.run(debug=True, port=5001) # Using port 5001 to avoid conflicts