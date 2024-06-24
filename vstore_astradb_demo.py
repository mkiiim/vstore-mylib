import os
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.ids import UUID
from astrapy.exceptions import InsertManyException

from vstore_config import *


# Initialize the client and get a "Database" object
def get_astra_client():
    client = DataAPIClient(ASTRA_TOKEN)
    return client

# Create a collection. The default similarity metric is cosine. If you're not
# sure what dimension to set, use whatever dimension vector your embeddings
# model produces.
def create_collection(database, namespace, name, dimension=5, metric=VectorMetric.COSINE):
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

    # create collection
    collection = create_collection(
        database=database,
        namespace='vectordb_docs',
        name='vector_test',
        dimension=6,
        metric=VectorMetric.COSINE
        ) 

    # Define documents
    # (UUIDs here are version 7.)
    documents = [
        {
            "_id": UUID("018e65c9-df45-7913-89f8-175f28bd7f74"),
            "text": "Chat bot integrated sneakers that talk to you",
            "dummy": "This is a dummy field",
            "$vector": [0.1, 0.15, 0.3, 0.12, 0.05, 0.1],
        },
        {
            "_id": UUID("018e65c9-e1b7-7048-a593-db452be1e4c2"),
            "text": "An AI quilt to help you sleep forever",
            "dummy": "This is a dummy field",
            "$vector": [0.45, 0.09, 0.01, 0.2, 0.11, 0.1],
        },
        {
            "_id": UUID("018e65c9-e33d-749b-9386-e848739582f0"),
            "text": "A deep learning display that controls your mood",
            "dummy": "This is a dummy field",
            "$vector": [0.1, 0.05, 0.08, 0.3, 0.6, 0.1],
        },
    ]

    # Insert the documents
    insert_documents(collection, documents)

    # Perform a similarity search
    query_vector = [0.15, 0.1, 0.1, 0.35, 0.55, 0.1]
    results = collection.find(
        sort={"$vector": query_vector},
        limit=10,
    )
    print("Vector search results:")
    for document in results:
        print("    ", document)

    # Cleanup (if desired)
    drop_result = collection.drop()
    print(f"\nCleanup: {drop_result}\n")

if __name__ == "__main__":
    main()