
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

st.set_page_config(
    page_title="Hybrid RAG Demo",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Hybrid Vector + Knowledge Graph RAG")
st.caption(
    "Ask questions about your PDF using semantic search "
    "and knowledge graph relationships."
)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "storage" / "chroma"


@st.cache_resource
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=key)


@st.cache_resource
def get_chroma_collection():
    db = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
    return db.get_collection("pdf_chunks")


@st.cache_resource
def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        raise ValueError("Neo4j settings are missing.")

    driver = GraphDatabase.driver(
        uri,
        auth=(username, password),
    )
    driver.verify_connectivity()
    return driver


def vector_search(question, limit=5):
    client = get_openai_client()
    collection = get_chroma_collection()

    embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        matches.append({
            "text": text,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return matches


def extract_query_entities(question):
    """Identify graph lookup terms from the question."""
    client = get_openai_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract up to 8 important named entities or "
                    "technical concepts explicitly mentioned in "
                    "the question. Return JSON only, using "
                    '{"entities": ["entity one", "entity two"]}. '
                    "Do not invent entities."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    import json

    data = json.loads(response.choices[0].message.content)
    return [
        value.strip()
        for value in data.get("entities", [])
        if isinstance(value, str) and value.strip()
    ][:8]


def graph_search(question):
    driver = get_neo4j_driver()
    terms = extract_query_entities(question)

    if not terms:
        return []

    query = """
    UNWIND $terms AS term
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower(term)
       OR toLower(term) CONTAINS toLower(e.name)
    WITH DISTINCT e
    OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
    RETURN e.name AS entity,
           type(r) AS relationship_type,
           r.relation AS relation,
           other.name AS related_entity,
           r.source AS source,
           r.page AS page
    LIMIT 30
    """

    with driver.session() as session:
        records = session.run(query, terms=terms)
        return [record.data() for record in records]


def answer_question(question, vector_results, graph_results):
    client = get_openai_client()

    vector_context = "\n\n".join(
        (
            f"[PDF: {item['metadata']['source']}, "
            f"page {item['metadata']['page']}]\n"
            f"{item['text']}"
        )
        for item in vector_results
    )

    graph_context = "\n".join(
        (
            f"Entity: {item['entity']}; "
            f"relationship: {item['relation']}; "
            f"related entity: {item['related_entity']}; "
            f"source: {item['source']}; page: {item['page']}"
        )
        for item in graph_results
        if item.get("related_entity")
    )

    if not vector_context and not graph_context:
        return (
            "I couldn't retrieve supporting evidence from "
            "the indexed PDF. Try rephrasing the question."
        )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a document-grounded RAG assistant.

Answer using only the supplied vector and graph evidence.
Do not invent facts or treat graph relationships as certain
if they conflict with the document passages.

Cite supporting passages using [filename, page N].
If the evidence is insufficient, explicitly say so.
Distinguish facts found in the PDF from inferences.
""",
            },
            {
                "role": "user",
                "content": f"""
QUESTION:
{question}

VECTOR SEARCH EVIDENCE:
{vector_context or "No vector evidence found."}

KNOWLEDGE GRAPH EVIDENCE:
{graph_context or "No graph evidence found."}

Provide a clear answer with source page citations.
""",
            },
        ],
    )

    return response.choices[0].message.content


with st.sidebar:
    st.header("System status")
    st.write("Vector store: ChromaDB")
    st.write("Knowledge graph: Neo4j Aura")
    st.write("Embeddings: text-embedding-3-small")
    st.write("LLM: gpt-4.1-mini")
    st.divider()
    st.caption("API usage may incur charges.")

question = st.chat_input("Ask a question about your PDF...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching both databases..."):
                vectors = vector_search(question)
                graph = graph_search(question)
                answer = answer_question(
                    question, vectors, graph
                )

            st.markdown(answer)

            with st.expander("Vector search results"):
                if not vectors:
                    st.write("No vector matches.")
                for item in vectors:
                    meta = item["metadata"]
                    st.markdown(
                        f"**{meta['source']} — "
                        f"page {meta['page']}**"
                    )
                    st.write(item["text"])
                    st.caption(
                        f"Distance: {item['distance']:.4f}"
                    )

            with st.expander("Knowledge graph results"):
                if not graph:
                    st.write("No graph relationships matched.")
                else:
                    st.json(graph)

        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.caption(
                "Check API credentials, ingestion status, "
                "and Neo4j connectivity."
            )
