
from pathlib import Path
import hashlib
import json

from chunker import chunk_text
from embedder import Embedder
from vector_store import VectorStore
from llm import LLM


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_DIR = PROJECT_ROOT / "data" / "policies"
DB_PATH = PROJECT_ROOT / "chroma_db_multi_policy"

METADATA_DIR = PROJECT_ROOT / "metadata"
HASH_FILE = METADATA_DIR / "policy_hashes.json"


# Map each policy filename (without .txt) to its category
POLICY_CATEGORIES = {
    "health_policy": "Health",
    "motor_policy": "Motor",
    "life_policy": "Life"
}


def load_policy(path):
    return path.read_text(encoding="utf-8")


# Calculate SHA-256 hash of policy content
def calculate_file_hash(path):

    file_content = path.read_bytes()

    return hashlib.sha256(file_content).hexdigest()


# Load previously saved policy hashes
def load_policy_hashes():

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not HASH_FILE.exists():
        return {}

    try:
        with HASH_FILE.open("r", encoding="utf-8") as file:
            hashes = json.load(file)

        if not isinstance(hashes, dict):
            raise ValueError("Hash metadata must be a JSON object.")

        return hashes

    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Could not read {HASH_FILE}. "
            "Please check or restore the metadata file."
        ) from error


# Save updated policy hashes
def save_policy_hashes(hashes):

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with HASH_FILE.open("w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=4)


def ingest_policies(embedder, vector_store):

    # Load previously stored hashes
    saved_hashes = load_policy_hashes()

    # Get all current policy files
    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    # Track current policy filenames
    current_sources = set()

    if not policy_files:
        print("No policy files found!")

    # Process each policy file
    for policy_path in policy_files:

        source = policy_path.stem

        if source not in POLICY_CATEGORIES:
            print(f"Skipping {source}: category not defined.")
            continue

        current_sources.add(source)

        category = POLICY_CATEGORIES[source]

        # Calculate current file hash
        current_hash = calculate_file_hash(policy_path)

        # Compare current hash with previously saved hash
        if saved_hashes.get(source) == current_hash:

            print(f"\nNo changes detected: {source}")
            print("Skipping ingestion.")
            continue

        print(f"\nNew or modified policy detected: {source}")
        print(f"Category: {category}")

        # Read the complete policy
        policy_text = load_policy(policy_path)

        # Split policy into chunks
        chunks = chunk_text(policy_text)

        print("Total chunks:", len(chunks))

        # Remove old chunks for this policy
        vector_store.delete_documents_by_source(source)

        if not chunks:
            print("No text chunks found.")
            saved_hashes[source] = current_hash
            continue

        # Generate embeddings
        embeddings = embedder.create_embeddings(chunks)

        print("Embeddings generated:", embeddings.shape)

        # Store updated chunks
        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            source=source,
            category=category
        )

        # Update hash only after successful storage
        saved_hashes[source] = current_hash

    # Detect deleted policy files
    deleted_sources = set(saved_hashes.keys()) - current_sources

    for source in deleted_sources:

        print(f"\nDeleted policy detected: {source}")

        # Remove its chunks from ChromaDB
        vector_store.delete_documents_by_source(source)

        # Remove its hash from metadata
        del saved_hashes[source]

    # Save updated hashes
    save_policy_hashes(saved_hashes)

    print("\nPolicy ingestion check completed!")


def run_rag(embedder, vector_store, llm, query, category=None):

    # Convert user's question into an embedding
    query_embedding = embedder.create_embeddings([query])[0]

    # Retrieve relevant policy chunks
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=4,
        category=category
    )

    print("\n===================================")
    print("User Question:", query)
    print("Category Filter:", category or "All Categories")
    print("===================================")

    if not results:
        print("No matching policy documents found.")
        return

    # Display retrieved context
    print("\nRETRIEVED POLICY CONTEXT")

    for result in results:

        print("\n---------------------------")
        print("Similarity Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Category:", result["category"])
        print("Chunk Index:", result["chunk_index"])
        print("Policy Text:", result["chunk"])

    # Combine retrieved chunks into context
    context = "\n\n".join(
        f"Source: {result['source']}\n"
        f"Category: {result['category']}\n"
        f"Policy Text: {result['chunk']}"
        for result in results
    )

    # Generate answer using LLM
    answer = llm.generate_answer(
        question=query,
        context=context
    )

    print("\n===================================")
    print("GENERATED ANSWER")
    print("===================================")
    print(answer)



def main():

    print("\n===================================")
    print("       POLICY QA ASSISTANT")
    print("===================================")

    # 1. Initialize embedding model ONCE
    embedder = Embedder()

    # 2. Initialize ChromaDB ONCE
    vector_store = VectorStore(DB_PATH)

    # 3. Initialize LLM ONCE
    llm = LLM()

    # 4. Check for new, modified or deleted policies
    print("\nChecking policy documents...")
    ingest_policies(embedder, vector_store)

    print("\nPolicy QA Assistant is ready!")
    print("Type 'exit' to close the application.")

    # 5. Continuous question-answering loop
    while True:

        print("\n-----------------------------------")

        query = input("\nEnter your policy question: ").strip()

        # Exit condition
        if query.lower() == "exit":
            print("\nThank you for using Policy QA Assistant!")
            break

        # Handle empty input
        if not query:
            print("Please enter a valid question.")
            continue

        # Ask for policy category
        category = input(
            "Enter category (Health/Motor/Life/All): "
        ).strip()

        if not category:
            category = "All"

        # Normalize category
        category = category.capitalize()

        if category == "All":
            category_filter = None

        elif category in ["Health", "Motor", "Life"]:
            category_filter = category

        else:
            print("Invalid category. Please try again.")
            continue

        # Generate answer
        run_rag(
            embedder,
            vector_store,
            llm,
            query=query,
            category=category_filter
        )


if __name__ == "__main__":
    main()
