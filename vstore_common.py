import os
import datetime
import openai
import numpy as np
import hashlib
import tiktoken
from collections import defaultdict

from vstore_config import *

# OpenAI
openai.api_key = APIKEY_OPENAI

# Calculate the hash of a file's content
def calculate_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

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
def sliding_window_wholewords(text, max_tokens=MAX_TOKENS_PER_FILE, overlap_words=OVERLAP_WORDS, model="text-embedding-ada-002", stop_at_chunk_index=None):
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

# Generate embeddings
def generate_vectors(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    elif not all(isinstance(text, str) and text for text in texts):
        raise ValueError("All elements in 'texts' must be non-empty strings.")
    response = openai.embeddings.create(model=model, input=texts)
    vectors = [np.array(data.embedding, dtype=np.float32) for data in response.data]
    return vectors