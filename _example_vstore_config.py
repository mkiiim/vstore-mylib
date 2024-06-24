import os
import json
import hashlib

# Directories containing the files to be indexed
DOCUMENT_DIR = f"/Users/{os.getenv('USER')}/MyProjects/vstore-mylib/docs/books"
VALID_FILE_TYPES = ['.txt', '.md']

# Hashed directory/collection for storing embeddings
EMBEDDING_DIR = "emb" + hashlib.md5(DOCUMENT_DIR.encode()).hexdigest()

# OpenAI
APIKEY_OPENAI = os.getenv('APIKEY_OPENAI')
EMBEDDING_ENGINE = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536
EMBEDDING_EMPTY_FILE = [[1e-9] * EMBEDDING_DIMENSION]

# Chunking Size
MAX_TOKENS_PER_FILE = 1024
OVERLAP_WORDS = 50

# Datastax AstraDB
ASTRA_DB_SCB = f"/Users/{os.getenv('USER')}/MyProjects/vstore-mylib/db/secure-connect-xxx.zip"
TOKEN_FILE = f"/Users/{os.getenv('USER')}/MyProjects/vstore-mylib/db/token.json"

ASTRA_DB_API_ENDPOINT = 'https://00000000-0000-0000-0000-000000000000-us-east1.apps.astra.datastax.com'
ASTRA_DB_ID = '00000000-0000-0000-0000-000000000000'
ASTRA_DB_REGION = 'us-east1'

with open(TOKEN_FILE, "r") as f:
    token = json.load(f)
    ASTRA_CLIENT_ID = token['clientId']
    ASTRA_CLIENT_SECRET = token['secret']
    ASTRA_TOKEN = token['token']

ASTRA_DB_NAMESPACE = 'vstore'

# FAISS
FAISS_INDEX_FILE = f"{EMBEDDING_DIR}/faiss_index.bin"
METADATA_FILE = f"{EMBEDDING_DIR}/metadata.pkl"


