
import numpy as np


class Retriever:

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query, embedder, top_k=2):

        # Convert the user's question into an embedding
        query_embedding = embedder.create_embeddings([query])[0]

        # Calculate cosine similarity
        similarities = np.dot(
            self.embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Get the indexes of the most relevant chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "chunk": self.chunks[index],
                "score": float(similarities[index])
            })

        return results
