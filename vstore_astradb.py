import os
import datetime
import numpy as np
import hashlib
from collections import defaultdict

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.ids import UUID
from astrapy.exceptions import InsertManyException

from vstore_common import *
from vstore_config import *

# Initialize the client and get a "Database" object
def get_astra_client():
    client = DataAPIClient(ASTRA_TOKEN)
    return client

# Create a collection. The default similarity metric is cosine. If you're not
# sure what dimension to set, use whatever dimension vector your embeddings
# model produces.
def create_collection(database, namespace, name, dimension=5, metric=VectorMetric.COSINE):

    # Check if the collection already exists
    if name in database.list_collection_names(namespace=namespace):
        print(f"* Collection already exists: {namespace}.{name}\n")
        collection = database.get_collection(namespace=namespace, name=name)
    else:
        print(f"* Creating collection: {namespace}.{name}")
        collection = database.create_collection(
            namespace=namespace,
            name=name,
            dimension=dimension,
            metric=metric,
            check_exists=False,
        )
        print(f"* Created collection: {collection.full_name}\n")
    return collection

# Read files from a directory and return a list of tuples containing the file path, file modified date, and file hash.
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
                    file_date = os.path.getmtime(filepath)
                    file_hash = calculate_hash(filepath)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # file_contents.append((filepath, f.read(), file_hash))
                        file_contents.append((filepath, file_date, file_hash))
    return file_contents

# Return files_processed
def process_files(directory, collection):
    file_contents = read_files(directory)

    # Create a set of file paths that are currently in the directory
    current_file_paths = set(filepath for filepath, _, _ in file_contents)

    # Create a set of file paths that are in the vector store collection
    collection_file_paths = set(entry['file_path'] for entry in collection.find())

    # Find the file paths that are in the metadata but not in the directory
    deleted_file_paths = collection_file_paths - current_file_paths
    print(f"* Found {len(deleted_file_paths)} deleted files.\n")

    # Remove the deleted file paths from the collection
    for file_path in deleted_file_paths:
        collection.delete_many({'file_path': file_path})

    # Update the set of file paths that are in the vector store collection
    collection_file_paths = set(entry['file_path'] for entry in collection.find())

    # Find the file paths that are in both the directory and the collection
    existing_file_paths = current_file_paths & collection_file_paths
    modified_file_paths = set()
    if existing_file_paths:
        # get the date of the most recent file in the collection
        most_recent_date = max(entry['last_updated'] for entry in collection.find())     
        # if any files in existing_file_paths have been modified since most_recent_date then add to modified_file_paths
        for file_path in existing_file_paths:
            file_date = os.path.getmtime(file_path)
            if file_date > most_recent_date:
                modified_file_paths.add(file_path)
                collection.delete_many({'file_path': file_path})
    print(f"* Found {len(modified_file_paths)} modified files to be removed and re-embedded.\n")

    # Find the file paths that are in the directory but not in the collection
    new_file_paths = current_file_paths - collection_file_paths
    print(f"* Found {len(new_file_paths)} new files.\n")

    # Return the file paths that need to be embedded
    files_to_embed = new_file_paths | modified_file_paths
    return files_to_embed

def embed_files(files_to_embed, collection):
    documents = []
    for file_index, file_path in enumerate(files_to_embed):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            hash_text = hashlib.md5(text.encode()).hexdigest()
            chunks = sliding_window_wholewords(text) if text != '' else  [''] # a chunk for an empty file
            for chunk_index, chunk in enumerate(chunks):
                print(f"\r* Processing file: {file_index+1} of {len(files_to_embed)}, Chunk Index: {chunk_index+1} of {len(chunks)}     ", end='', flush=True)
                hash_chunk = hashlib.md5(chunk.encode()).hexdigest()
                vector = generate_vectors(chunk) if chunk != '' else EMBEDDING_EMPTY_FILE # for cases when can't have zero vectors e.g. for cosine similarity
                documents.append({
                    "file_path": file_path,
                    "chunk_index": chunk_index,
                    "chunk": chunk,
                    "hash_text": hash_text,
                    "hash_chunk": hash_chunk,
                    "last_updated": datetime.datetime.now().timestamp(),
                    "$vector": vector[0]
                })
    print("\n")
    return documents

# Insert documents into the collection.
def insert_documents(collection, documents):
    print(f"* Inserting {len(documents)} documents...")
    try:
        insertion_result = collection.insert_many(documents)
        print(f"* Inserted {len(insertion_result.inserted_ids)} items.\n")
        return insertion_result
    except InsertManyException:
        print("* Documents found on DB already. Let's move on.\n")

def main():

    # initialize the Astra client
    client = get_astra_client()
    database = client.get_database(ASTRA_DB_API_ENDPOINT)
    print(f"* Database: {database.info().name}\n")

    # create/get collection
    collection = create_collection(
        database=database,
        namespace=ASTRA_DB_NAMESPACE,
        name=EMBEDDING_DIR,
        dimension=EMBEDDING_DIMENSION,
        metric=VectorMetric.COSINE
        ) 

    # Generate list of files requiring embedding
    files_to_embed = process_files(DOCUMENT_DIR, collection)

    # Embed the files
    if files_to_embed:
        documents = embed_files(files_to_embed, collection)

        # Insert the documents
        if documents:
            insert_documents(collection, documents)

    # Perform a similarity search
    query_text = "The quality of a man's time spent on earth."
    query_vector = generate_vectors([query_text])[0]
    results = collection.find(
        sort={"$vector": query_vector},
        limit=5,
    )
    print("Vector search results:")
    for document in results:
        print("    ", document)

    # # Cleanup (if desired)
    # drop_result = collection.drop()
    # print(f"\nCleanup: {drop_result}\n")

if __name__ == "__main__":
    main()