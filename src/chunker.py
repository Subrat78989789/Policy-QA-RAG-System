
from typing import List

def chunk_text(text: str) -> List[str]:
    """
    Split policy text into smaller chunks.
    Each paragraph becomes one chunk.
    """

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks
