# image_search_app.py

import os
import json
import pickle  # Used for saving/loading complex Python objects like the index
import time
import sys

# --- Assuming your A1 files are accessible ---
# Adjust path if they are in a subdirectory like 'src'
# Example: from src.preprocessor import Preprocessor
try:
    from preprocessor import Preprocessor
    from indexer import Indexer
    from search import SearchEngine
except ImportError as e:
    print(f"Error importing assignment 1 modules: {e}")
    print("Make sure preprocessor.py, indexer.py, and search.py are in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# --- Configuration ---
METADATA_FILE = 'fandom_image_data/fandom_image_metadata.json'
INDEX_DIR = 'image_index_data' # Directory to store saved index files
INDEX_COMPONENTS_FILE = os.path.join(INDEX_DIR, 'index_components.pkl')
DOC_ID_MAP_FILE = os.path.join(INDEX_DIR, 'doc_id_map.json')

# Create index directory if it doesn't exist
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Index Building Function ---
def build_index_from_json(json_file_path):
    """Loads image metadata, builds the index using A1 code, and returns indexer + mapping."""
    print(f"Loading image metadata from {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            image_data = json.load(f)
        print(f"Loaded {len(image_data)} image metadata entries.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {json_file_path}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None, None

    # Initialize components from Assignment 1
    preprocessor = Preprocessor()
    indexer = Indexer(preprocessor) # Pass the preprocessor to the indexer

    # Create mapping from our simple sequential doc_id back to image details
    doc_id_to_metadata = {}

    print("Starting indexing process...")
    start_time = time.time()

    # Loop through metadata and index
    for i, item in enumerate(image_data):
        doc_id = i # Use simple integer ID (0, 1, 2...)
        content = item.get('context', '') # Get the text surrogate
        image_url = item.get('image_url')
        source_page = item.get('source_page')

        if not content or not image_url:
            print(f"Warning: Skipping item {i} due to missing context or image_url.")
            continue

        # Store mapping from internal ID back to important details
        doc_id_to_metadata[doc_id] = {
            'image_url': image_url,
            'source_page': source_page,
            'alt_text': item.get('alt_text', '') # Store alt text too if needed
        }

        # Add document to the indexer (using the A1 method)
        indexer.add_document(doc_id, content)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_data)} documents...")

    print(f"Initial indexing phase complete. Processed {indexer.doc_count} valid documents.")

    # Finalize the index (calculates avg doc length, builds VSM vectors)
    print("Finalizing index (calculating stats and VSM vectors)...")
    indexer.finalize_index()

    end_time = time.time()
    print(f"Indexing finished in {end_time - start_time:.2f} seconds.")
    print(f"  Vocabulary size: {len(indexer.inverted_index)} terms")
    print(f"  Average document length: {indexer.avg_doc_length:.2f} terms")

    return indexer, doc_id_to_metadata

# --- Save/Load Functions ---
def save_index_data(indexer, doc_id_map, index_file, map_file):
    """Saves the core index data structures and the ID map."""
    print(f"Saving index components to {index_file}...")
    try:
        # Save the necessary data structures from the indexer
        index_data_to_save = {
            'inverted_index': indexer.inverted_index,
            'document_lengths': indexer.document_lengths,
            'doc_count': indexer.doc_count,
            'avg_doc_length': indexer.avg_doc_length,
            'document_vectors': indexer.document_vectors, # For VSM
            'document_norms': indexer.document_norms      # For VSM
        }
        with open(index_file, 'wb') as f:
            pickle.dump(index_data_to_save, f)
        print("Index components saved successfully.")
    except Exception as e:
        print(f"Error saving index components: {e}")
        return False

    print(f"Saving document ID mapping to {map_file}...")
    try:
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(doc_id_map, f, indent=4)
        print("Document ID mapping saved successfully.")
        return True
    except IOError as e:
        print(f"Error saving mapping file: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during mapping save: {e}")
        return False


def load_index_data(index_file, map_file):
    """Loads the index data structures and ID map."""
    print(f"Loading index components from {index_file}...")
    try:
        with open(index_file, 'rb') as f:
            loaded_index_data = pickle.load(f)
        print("Index components loaded.")
    except FileNotFoundError:
        print("Error: Index components file not found. Please build the index first.")
        return None, None
    except Exception as e:
        print(f"Error loading index components: {e}")
        return None, None

    print(f"Loading document ID mapping from {map_file}...")
    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            doc_id_map = json.load(f)
            # Convert JSON string keys back to integers if needed (depends on how you saved it)
            # If you used integer keys when building/saving, JSON saves them as strings.
            doc_id_map = {int(k): v for k, v in doc_id_map.items()}
        print("Document ID mapping loaded.")
    except FileNotFoundError:
        print("Error: Document ID mapping file not found. Please build the index first.")
        return None, None
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return None, None

    # Reconstruct the Indexer object
    try:
        preprocessor = Preprocessor() # Need a preprocessor for the SearchEngine later
        reconstructed_indexer = Indexer(preprocessor)

        # Populate the reconstructed indexer with loaded data
        reconstructed_indexer.inverted_index = loaded_index_data['inverted_index']
        reconstructed_indexer.document_lengths = loaded_index_data['document_lengths']
        reconstructed_indexer.doc_count = loaded_index_data['doc_count']
        reconstructed_indexer.avg_doc_length = loaded_index_data['avg_doc_length']
        reconstructed_indexer.document_vectors = loaded_index_data['document_vectors']
        reconstructed_indexer.document_norms = loaded_index_data['document_norms']

        print("Indexer object reconstructed successfully.")
        return reconstructed_indexer, doc_id_map, preprocessor # Return preprocessor too
    except Exception as e:
        print(f"Error reconstructing indexer object: {e}")
        return None, None, None


# --- Search Function ---
def perform_search(query_text, search_engine, model_name, doc_id_map, top_k=10):
    """Performs search using the chosen model and returns formatted results."""
    print(f"\nSearching for '{query_text}' using {model_name}...")
    start_time = time.time()

    results = []
    if model_name == 'vsm':
        # Assuming top_k=100 was default in search_vsm, adjust if needed
        raw_results = search_engine.search_vsm(query_text, top_k=top_k)
    elif model_name == 'bm25':
        raw_results = search_engine.search_bm25(query_text, top_k=top_k)
    elif model_name == 'lm_dirichlet':
        raw_results = search_engine.search_lm_dirichlet(query_text, top_k=top_k)
    else:
        print(f"Error: Unknown model '{model_name}'")
        return []

    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds.")

    if not raw_results:
        print("No results found.")
        return []

    # Format results using the doc_id_map
    for rank, (doc_id, score) in enumerate(raw_results, 1):
        metadata = doc_id_map.get(doc_id) # Use .get for safety
        if metadata:
            results.append({
                'rank': rank,
                'score': score,
                'image_url': metadata['image_url'],
                'source_page': metadata['source_page']
                # 'alt_text': metadata.get('alt_text', '') # Optionally include alt text
            })
        else:
            print(f"Warning: Could not find metadata for doc_id {doc_id}")

    return results

# --- Main Execution ---
if __name__ == "__main__":
    # --- Step 1: Build or Load Index ---
    indexer = None
    doc_id_map = None
    preprocessor = None

    if os.path.exists(INDEX_COMPONENTS_FILE) and os.path.exists(DOC_ID_MAP_FILE):
        print("Found existing index files.")
        action = input("Load existing index (L) or Rebuild index (R)? ").strip().upper()
        if action == 'L':
            indexer, doc_id_map, preprocessor = load_index_data(INDEX_COMPONENTS_FILE, DOC_ID_MAP_FILE)
        elif action == 'R':
            indexer, doc_id_map = build_index_from_json(METADATA_FILE)
            if indexer and doc_id_map:
                save_index_data(indexer, doc_id_map, INDEX_COMPONENTS_FILE, DOC_ID_MAP_FILE)
                # After saving and potentially rebuilding, we still need a preprocessor
                preprocessor = Preprocessor()
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    else:
        print("No existing index found. Building index...")
        indexer, doc_id_map = build_index_from_json(METADATA_FILE)
        if indexer and doc_id_map:
            save_index_data(indexer, doc_id_map, INDEX_COMPONENTS_FILE, DOC_ID_MAP_FILE)
            preprocessor = Preprocessor() # Need preprocessor after building

    # Exit if index loading/building failed
    if not indexer or not doc_id_map or not preprocessor:
        print("Failed to initialize indexer. Exiting.")
        sys.exit(1)

    # --- Step 2: Initialize Search Engine ---
    search_engine = SearchEngine(indexer, preprocessor) # Pass loaded/built indexer and preprocessor
    print("\nSearch engine initialized.")

    # --- Step 3: Interactive Search Loop ---
    while True:
        print("\n--- Image Search ---")
        print("Available models: vsm, bm25, lm_dirichlet")
        model = input("Choose search model (or type 'quit'): ").strip().lower()

        if model == 'quit':
            break
        elif model not in ['vsm', 'bm25', 'lm_dirichlet']:
            print("Invalid model selected. Please try again.")
            continue

        query = input(f"Enter query for {model}: ").strip()
        if not query:
            continue

        # Perform search and get formatted results
        search_results = perform_search(query, search_engine, model, doc_id_map)

        # Display results
        if search_results:
            print(f"\nTop {len(search_results)} results for '{query}' ({model}):")
            for res in search_results:
                print(f"  Rank: {res['rank']}")
                print(f"  Score: {res['score']:.4f}")
                print(f"  Image: {res['image_url']}")
                print(f"  Source: {res['source_page']}")
                print("-" * 20)
        # perform_search already prints "No results found." if empty

    print("\nExiting image search.")