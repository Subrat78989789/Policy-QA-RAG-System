
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully!")

    def create_embeddings(self, chunks: list[str]):
        """
        Convert text chunks into numerical vectors.
        """
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True
        )

        return embeddings
