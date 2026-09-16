"""Chunk the policy markdown docs and load them into policy_chunks.

Reads support_agent/data/policies/*.md, splits each on its `## N. Section
Name` headings, embeds each chunk's content with all-MiniLM-L6-v2, and
replaces the table contents in one transaction. Re-run any time the policy
docs change; there is no incremental/partial-update path -- this always
rebuilds the whole table from the current files on disk.
"""

import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from shared.database import get_session
from shared.models import PolicyChunk

POLICIES_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text on `## ` headings into (section, content) pairs.

    Each section's content runs to the start of the next heading, or to
    the end of the text for the last one -- there is no dependency on a
    terminating marker, so the final section is never dropped or cut
    short.
    """
    headings = list(HEADING_RE.finditer(text))
    sections = []
    for i, match in enumerate(headings):
        section = match.group(1)
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()
        sections.append((section, content))
    return sections


def load_chunks() -> list[dict]:
    """Read every policy doc and return raw chunk dicts (no embeddings yet)."""
    chunks = []
    for path in sorted(POLICIES_DIR.glob("*.md")):
        document = path.stem
        text = path.read_text()
        for section, content in parse_sections(text):
            chunks.append({"document": document, "section": section, "content": content})
    return chunks


def main() -> None:
    chunks = load_chunks()

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["content"] for c in chunks])

    with get_session() as session:
        session.execute(text("TRUNCATE TABLE policy_chunks RESTART IDENTITY"))
        session.add_all(
            PolicyChunk(
                document=chunk["document"],
                section=chunk["section"],
                content=chunk["content"],
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        )
        session.commit()

    per_document = {}
    for chunk in chunks:
        per_document[chunk["document"]] = per_document.get(chunk["document"], 0) + 1

    print("Chunks per document:")
    for document, count in sorted(per_document.items()):
        print(f"  {document}: {count}")
    print(f"Total chunks: {len(chunks)}")

    example, example_embedding = chunks[0], embeddings[0]
    print()
    print(f"Example chunk: [{example['document']} / {example['section']}]")
    print(f"  content[:80]: {example['content'][:80]!r}")
    print(f"  embedding[:5]: {example_embedding[:5].tolist()}")


if __name__ == "__main__":
    main()
