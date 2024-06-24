import os
import openai
import numpy as np
import faiss
import pickle
import hashlib
import tiktoken
from vstore_config import *
from collections import defaultdict

# Set your OpenAI API key
openai.api_key = APIKEY_OPENAI

# Directory to process
directory_to_process = DOCUMENT_DIR

# Calculate the hash of a file's content
def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def read_files(directory):
    file_contents = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            # Check if the file has a valid extension
            if any(file.endswith(ext) for ext in VALID_FILE_TYPES):
                filepath = os.path.join(root, file)
                if not file.startswith('.') and os.path.isfile(filepath):
                    file_hash = calculate_hash(filepath)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_contents.append((filepath, f.read(), file_hash))
    return file_contents

# Split text - OLD, not used
def split_text(text, max_tokens=2048):
    tokens = text.split()
    return [' '.join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

# Split text using a sliding window approach - OLD, not used
def sliding_window(text, max_tokens=2048, overlap=200, model="text-embedding-ada-002"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

# Split text using a sliding window approach with whole words
def sliding_window_wholewords(text, max_tokens=2048, overlap_words=150, model="text-embedding-ada-002", stop_at_chunk_index=None):
    enc = tiktoken.encoding_for_model(model)
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        word_tokens = len(enc.encode(word))
        current_chunk.append(word)
        current_tokens += word_tokens
        if current_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap_words:]
            current_tokens = sum(len(enc.encode(word)) for word in current_chunk)
            if stop_at_chunk_index is not None and len(chunks) > stop_at_chunk_index:
                break
    if current_chunk and (stop_at_chunk_index is None or len(chunks) <= stop_at_chunk_index):
        chunks.append(' '.join(current_chunk))
    return chunks

# Count tokens using tiktoken
def count_tokens(text, model="text-embedding-ada-002"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Generate vectors 
def generate_vectors(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    elif not all(isinstance(text, str) and text for text in texts):
        raise ValueError("All elements in 'texts' must be non-empty strings.")
    response = openai.embeddings.create(model=model, input=texts)
    vectors = [np.array(data.embedding, dtype=np.float32) for data in response.data]
    return vectors

# Create a new FAISS index
def create_faiss_index(dim):
    return faiss.IndexFlatL2(dim)

# Save FAISS index and metadata to disk
def write_file_faiss_index_and_metadata(faiss_index, metadata, index_file=FAISS_INDEX_FILE, metadata_file=METADATA_FILE):
    
    # Ensure the directory for the index file exists
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    # Ensure the directory for the metadata file exists
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    faiss.write_index(faiss_index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def write_db_faiss_index_and_metadata():
    pass

# Load FAISS index and metadata from disk
def read_file_faiss_index_and_metadata(index_file=FAISS_INDEX_FILE, metadata_file=METADATA_FILE):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        faiss_index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        faiss_index = None
        metadata = []
    return faiss_index, metadata

def read_db_faiss_index_and_metadata():
    pass


# Return files_processed
def process_and_add_files(directory, faiss_index, metadata):
    file_contents = read_files(directory)
    files_processed = False

    # Create a set of file paths that are currently in the directory
    current_file_paths = set(filepath for filepath, _, _ in file_contents)

    # Create a set of file paths that are in the metadata
    metadata_file_paths = set(entry['file_path'] for entry in metadata)

    # Find the file paths that are in the metadata but not in the directory
    deleted_file_paths = metadata_file_paths - current_file_paths

    # For each deleted file path
    for deleted_file_path in deleted_file_paths:
        # Find the corresponding entries in the metadata
        deleted_entries = [entry for entry in metadata if entry['file_path'] == deleted_file_path]

        # For each deleted entry
        for deleted_entry in deleted_entries:
            # Remove the vector from the FAISS index
            old_idx = metadata.index(deleted_entry)
            faiss_index.remove_ids(np.array([old_idx]))

            # Remove the entry from the metadata
            metadata.pop(old_idx)

        files_processed = True

    for file_index, (filepath, content, file_hash) in enumerate(file_contents, start=1):
        # Check if the file is already processed
        existing_entries = [entry for entry in metadata if entry['file_path'] == filepath]
        if existing_entries:
            # Update if the hash has changed
            if existing_entries[0]['file_hash'] == file_hash:
                continue  # No changes, skip this file
            else:
                # Remove the old vectors if the file has been modified
                for existing_entry in existing_entries:
                    old_vector = existing_entry['vector']
                    old_idx = metadata.index(existing_entry)
                    faiss_index.remove_ids(np.array([old_idx]))
                    metadata.pop(old_idx)

        # Split and process text
        if count_tokens(content) > MAX_TOKENS_PER_FILE:
            chunks = sliding_window_wholewords(content, MAX_TOKENS_PER_FILE)
        else:
            chunks = [content]

        # Generate vectors for chunks and add to index
        for chunk_index, chunk in enumerate(chunks):
            print(f"\rProcessing file: {file_index+1} of {len(file_contents)}, Chunk Index: {chunk_index+1} of {len(chunks)}", end='', flush=True)
            if chunk == '':
                vector = np.zeros((EMBEDDING_DIMENSION,), dtype=np.float32)  # Placeholder vector for empty files
            else:
                vector = generate_vectors([chunk])[0]

            faiss_index.add(np.array([vector]))
            metadata.append({'file_path': filepath, 'file_hash': file_hash, 'vector': vector, 'chunk_index': chunk_index})

        files_processed = True
        print()
    return files_processed

# Search FAISS index
def search_faiss_index(faiss_index, query_vector, k=5):
    distances, indices = faiss_index.search(np.array([query_vector]), k)
    return distances, indices

# # Search FAISS index for the most similar files to a query
# def search_faiss_index_files(faiss_index, query_vector, k=5):
#     distances, indices = faiss_index.search(np.array([query_vector]), k)
#     file_paths = []
#     for i in indices[0]:
#         file_paths.append(metadata[i]['file_path'])
#     return file_paths

# # Search FAISS index for the most similar chunks to a query
# def search_faiss_index_chunks(faiss_index, query_vector, k=5):
#     distances, indices = faiss_index.search(np.array([query_vector]), k)
#     chunks = []
#     for i in indices[0]:
#         chunks.append({'file_path': metadata[i]['file_path'], 'chunk_index': metadata[i]['chunk_index']})
#     return chunks

# Execute query and print results
def execute_query_and_print_results(faiss_index, metadata, query_text):
    query_vector = generate_vectors([query_text])[0]
    distances, indices = search_faiss_index(faiss_index, query_vector)
    
    # Group results by file and find the highest chunk index for each file
    results_by_file = defaultdict(list)
    for dist, idx in zip(distances[0], indices[0]):
        file_path = metadata[idx]['file_path']
        chunk_index = metadata[idx]['chunk_index']
        results_by_file[file_path].append((chunk_index, dist, idx))

    # Generate chunks for all files
    chunks_by_file = {}
    for file_path, results in results_by_file.items():
        # Find the highest chunk index for this file
        highest_chunk_index = max(chunk_index for chunk_index, _, _ in results)
        
        # Generate chunks up to the highest chunk index
        with open(file_path, 'r') as file:
            content = file.read()
        chunks_by_file[file_path] = sliding_window_wholewords(content, MAX_TOKENS_PER_FILE, stop_at_chunk_index=highest_chunk_index)

    print(f"\nQuery: {query_text}\n")
    print("Top results:")
    # Print the results in the original order
    for dist, idx in zip(distances[0], indices[0]):
        file_path = metadata[idx]['file_path']
        chunk_index = metadata[idx]['chunk_index']
        chunk_text = chunks_by_file[file_path][chunk_index]
        print(f"Rank: {indices[0].tolist().index(idx) + 1}")
        print(f"File: {file_path}, Chunk Index: {chunk_index}, Distance: {dist}")
        print(f"Chunk Text: {chunk_text[:160]}\n")

def main():
    # Load existing metadata and FAISS index if available
    faiss_index, metadata = read_file_faiss_index_and_metadata()
    
    if faiss_index is None:
        # Dummy text to get the dimension if no metadata is available
        dummy_text = "This is a dummy text to initialize the FAISS index dimension."
        dummy_vector = generate_vectors([dummy_text])[0]
        dim = len(dummy_vector)
        faiss_index = create_faiss_index(dim)
    
    # Track if any files were processed
    files_processed = False
    
    # Process and add files to the index
    files_processed = process_and_add_files(directory_to_process, faiss_index, metadata)
    
    # Save FAISS index and metadata for future use only if files were processed
    if files_processed:
        write_file_faiss_index_and_metadata(faiss_index, metadata)
        print("Vector store created/updated in FAISS.")
    else:
        print("No new files processed. Vector store not updated.")
    
    # Example query
    query_text = "Man is silly and petulant most of the time."
    execute_query_and_print_results(faiss_index, metadata, query_text)

if __name__ == "__main__":
    main()