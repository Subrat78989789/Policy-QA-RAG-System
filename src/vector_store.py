import chromadb
from pathlib import Path


class VectorStore:

    def __init__(self, persist_directory):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="policy_documents",
            metadata={"hnsw:space": "cosine"}
        )

        print("ChromaDB initialized successfully!")


    def add_documents(self, chunks, embeddings, source, category):

        ids = [
            f"{source}_{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source,
                "category": category,
                "chunk_index": index
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"Stored {len(chunks)} chunks from "
            f"{source} in category {category}!"
        )

    def delete_documents_by_source(self, source):
        self.collection.delete(
            where={"source": source}
        )
        print(f"Deleted existing chunks for: {source}")

    def search(self, query_embedding, top_k=2, category=None):

        total_documents = self.collection.count()

        if total_documents == 0:
            print("No documents found in ChromaDB.")
            return []

        query_args = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, total_documents),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        # Apply category filtering when a category is provided
        if category:
            query_args["where"] = {
                "category": category
            }

        results = self.collection.query(**query_args)

        retrieved_documents = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            retrieved_documents.append({
                "chunk": document,
                "source": metadata["source"],
                "category": metadata["category"],
                "chunk_index": metadata["chunk_index"],
                "score": 1 - distance
            })


        return retrieved_documents