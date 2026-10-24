
import os
import json
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
CHROMA_DIR = BASE_DIR / "storage" / "chroma"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def make_id(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def split_text(text, chunk_size=1200, overlap=200):
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def extract_pdf_chunks(pdf_path):
    """Extract page text and preserve page numbers."""
    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for chunk_number, chunk in enumerate(
            split_text(text), start=1
        ):
            chunk_id = make_id(
                f"{pdf_path.name}:{page_number}:"
                f"{chunk_number}:{chunk}"
            )

            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "source": pdf_path.name,
                "page": page_number,
                "chunk": chunk_number,
            })

    return chunks


def create_embeddings(texts):
    """Generate vectors in batches."""
    vectors = []

    for start in range(0, len(texts), 50):
        batch = texts[start:start + 50]

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        vectors.extend(
            item.embedding for item in response.data
        )

    return vectors


def store_in_chroma(chunks):
    """Persist text chunks and embeddings."""
    db = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = db.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    # Remove previous entries for this PDF before re-ingesting.
    # This prevents stale chunks from surviving document updates.
    existing = collection.get(
        where={"source": chunks[0]["source"]},
        include=[],
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    texts = [item["text"] for item in chunks]
    embeddings = create_embeddings(texts)

    collection.upsert(
        ids=[item["id"] for item in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": item["source"],
                "page": item["page"],
                "chunk": item["chunk"],
            }
            for item in chunks
        ],
    )

    return len(chunks)


def extract_graph_data(text):
    """Extract candidate entities and relationships."""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Extract explicit entities and relationships from
the supplied document passage.

Return JSON with this structure:
{
  "entities": ["Entity A", "Entity B"],
  "relationships": [
    {
      "source": "Entity A",
      "relation": "depends_on",
      "target": "Entity B"
    }
  ]
}

Rules:
- Use only facts explicitly supported by the passage.
- Do not invent entities or relationships.
- Keep entity names concise and consistent.
- Use a short snake_case relation name.
- Return empty arrays when nothing is identifiable.
"""
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    return json.loads(
        response.choices[0].message.content
    )


def store_in_neo4j(chunks):
    """Store entities and relationships in Neo4j."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        raise ValueError(
            "Missing Neo4j settings in .env"
        )

    driver = GraphDatabase.driver(
        uri, auth=(username, password)
    )

    entity_count = 0
    relationship_count = 0

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            for item in chunks:
                graph = extract_graph_data(item["text"])

                # Store entities as graph nodes.
                names = set(graph.get("entities", []))

                for rel in graph.get("relationships", []):
                    names.add(rel["source"])
                    names.add(rel["target"])

                for name in names:
                    if not isinstance(name, str) or not name.strip():
                        continue

                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        """,
                        name=name.strip(),
                    )
                    entity_count += 1

                # Store typed relationships using a safe,
                # fixed Neo4j relationship type.
                for rel in graph.get("relationships", []):
                    source = rel.get("source", "").strip()
                    target = rel.get("target", "").strip()
                    relation = rel.get("relation", "").strip()

                    if not source or not target or not relation:
                        continue

                    session.run(
                        """
                        MERGE (a:Entity {name: $source})
                        MERGE (b:Entity {name: $target})
                        MERGE (a)-[r:RELATED_TO {
                            relation: $relation,
                            chunk_id: $chunk_id
                        }]->(b)
                        SET r.source = $document,
                            r.page = $page
                        """,
                        source=source,
                        target=target,
                        relation=relation,
                        chunk_id=item["id"],
                        document=item["source"],
                        page=item["page"],
                    )
                    relationship_count += 1

    finally:
        driver.close()

    return entity_count, relationship_count


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF found in {PDF_DIR}"
        )

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")
        chunks = extract_pdf_chunks(pdf_path)

        if not chunks:
            print(f"No extractable text in {pdf_path.name}")
            continue

        print(f"Extracted {len(chunks)} chunks")

        store_in_chroma(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(
            "No text chunks were extracted from the PDFs."
        )

    print("Extracting knowledge graph relationships...")
    entities, relationships = store_in_neo4j(all_chunks)

    print("\nIngestion completed.")
    print(f"Chunks processed: {len(all_chunks)}")
    print(f"Entity records processed: {entities}")
    print(f"Relationship records processed: {relationships}")


if __name__ == "__main__":
    main()
