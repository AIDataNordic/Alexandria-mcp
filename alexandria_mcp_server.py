"""
Alexandria MCP Server - fastmcp v3
Semantic search over 4.6M+ classical philosophy and humanities texts from Archive.org,
with hybrid dense+sparse retrieval and cross-encoder reranking.

Collection: alexandria
Vectors:    4.6M+ chunks from 13,000+ books
Embedding:  intfloat/multilingual-e5-large (1024 dim)
Reranker:   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
"""

import os
import logging
import time
import torch
from typing import Annotated
from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector,
    Filter, FieldCondition, MatchText, MatchValue,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

# --- Configuration ---
COLLECTION_NAME = "alexandria"
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
RERANK_FETCH    = 40
RERANK_MODEL    = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print("Loading embedding model...")
_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
_model.max_seq_length = 512
print("Embedding model loaded.")

print("Loading sparse model...")
_sparse_model = SparseTextEmbedding("Qdrant/bm25")
print("Sparse model loaded.")

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL, device="cpu")
print("Reranker loaded.")

_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Logging ---
_log = logging.getLogger("alexandria_mcp")
_log.setLevel(logging.INFO)
os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)
_fh = logging.FileHandler(os.path.expanduser("~/logs/alexandria_mcp_server.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
_log.addHandler(_fh)

mcp = FastMCP("alexandria-philosophy-mcp")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def ping(name: Annotated[str, "Name to greet"] = "world") -> str:
    """Simple connectivity test. Returns a greeting to confirm the server is running."""
    _log.info(f'ping name="{name}"')
    return f"Hello {name}! Alexandria MCP server is running with 4.6M+ philosophy texts."


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def search_texts(
    query: Annotated[str, "What you are looking for, e.g. 'Nietzsche will to power', 'Kantian categorical imperative', 'Platonic theory of forms', 'Stoic virtue and the sage'"],
    author: Annotated[str, "Filter results to a specific author/creator, e.g. 'Kant', 'Nietzsche', 'Aristotle'. Case-insensitive substring match."] = "",
    language: Annotated[str, "Filter by language code: 'eng', 'ger', 'lat', 'fre', 'ita', 'gre', 'rus'"] = "",
    limit: Annotated[int, "Number of results after reranking (default 5, max 20)"] = 5,
) -> list[dict]:
    """Search 4.6 million classical philosophy and humanities texts from Archive.org.

    The collection contains public domain books (pre-1928) covering:
      - Philosophy: Aristotle, Plato, Kant, Hegel, Nietzsche, Schopenhauer,
        Descartes, Spinoza, Locke, Hume, Mill, Wittgenstein, Aquinas and many more
      - Ethics, metaphysics, epistemology, logic, political philosophy
      - Sacred and religious texts, stoicism, neoplatonism, existentialism
      - Classical literature, history of ideas, social theory
      - Sources: Internet Archive (americana, europeanlibraries, gutenberg)

    Texts are in original languages — primarily English, German, Latin, French,
    Italian, Greek, Russian. Queries in any language work due to multilingual embeddings.

    Args:
        query:    What you are looking for, e.g. 'Nietzsche will to power eternal recurrence',
                  'Kantian categorical imperative duty ethics',
                  'Platonic theory of forms and the Good',
                  'Stoic virtue and the sage', 'Aristotle eudaimonia flourishing',
                  'Hegel dialectics spirit history', 'free will determinism compatibilism'
        author:   Optional — filter results to a specific author/creator,
                  e.g. 'Kant', 'Nietzsche', 'Aristotle'. Case-insensitive substring match.
        language: Optional — filter by language code, e.g. 'eng', 'ger', 'lat',
                  'fre', 'ita', 'gre', 'rus'
        limit:    Number of results after reranking (default 5, max 20)

    Returns:
        List of relevant text excerpts with metadata, reranked by relevance.
        Each result includes rerank_score, vector_score, title, creator,
        date, language, subject and the full text chunk.
    """
    limit = min(limit, 20)
    _t0 = time.time()

    # multilingual-e5-large uses "query:" prefix at search time
    e5_query = f"query: {query}"

    with torch.no_grad():
        dense_vec = _model.encode(e5_query, normalize_embeddings=True).tolist()

    sparse_result = list(_sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    qfilter = Filter(must=[FieldCondition(key="creator", match=MatchText(text=author))]) if author else None

    fetch_limit = max(RERANK_FETCH, limit * 4)
    try:
        results = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec,  using="dense",  limit=fetch_limit),
                Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qfilter,
            limit=fetch_limit,
            with_payload=True,
        )
    except Exception as e:
        _log.exception(f"Qdrant query failed: {e}")
        return []

    if not results.points:
        return []

    # Filter by language if specified
    candidates = results.points
    if language:
        candidates = [
            p for p in candidates
            if language.lower() in (
                [p.payload.get("language")] if isinstance(p.payload.get("language"), str)
                else (p.payload.get("language") or [])
            )
        ]
        if not candidates:
            candidates = results.points  # fall back if filter yields nothing

    pairs = [(query, p.payload.get("text", "")) for p in candidates]
    with torch.no_grad():
        rerank_scores = _reranker.predict(pairs)

    r_scores = [float(s) for s in rerank_scores]
    v_scores = [p.score for p in candidates]
    v_max = max(v_scores) or 1e-8

    hybrid_scores = [
        r * (1.0 + 0.3 * (v / v_max))
        for r, v in zip(r_scores, v_scores)
    ]

    ranked = sorted(
        zip(hybrid_scores, r_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    output = []
    for hybrid_score, rerank_score, point in ranked:
        if len(output) >= limit:
            break
        p = point.payload
        subject = p.get("subject", "")
        if isinstance(subject, list):
            subject = ", ".join(subject[:3])
        output.append({
            "rerank_score": round(float(rerank_score), 4),
            "hybrid_score": round(float(hybrid_score), 4),
            "vector_score": round(point.score, 4),
            "title":        p.get("title"),
            "creator":      p.get("creator"),
            "date":         p.get("date", "")[:4] if p.get("date") else None,
            "language":     p.get("language"),
            "subject":      subject,
            "identifier":   p.get("identifier"),
            "chunk_index":  p.get("chunk_index"),
            "total_chunks": p.get("total_chunks"),
            "text":         p.get("text"),
        })

    elapsed = round(time.time() - _t0, 3)
    _log.info(f'search_texts query="{query}" author="{author}" language="{language}" results={len(output)} elapsed={elapsed}s')
    return output


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
async def get_book_list(
    author: Annotated[str, "Filter by author/creator name, e.g. 'Kant', 'Nietzsche', 'Plato'. Case-insensitive substring match."] = "",
    subject: Annotated[str, "Filter by subject keyword, e.g. 'ethics', 'logic', 'metaphysics'. Case-insensitive substring match."] = "",
    language: Annotated[str, "Filter by language code: 'eng', 'ger', 'lat', 'fre', 'ita', 'gre', 'rus'"] = "",
    limit: Annotated[int, "Maximum number of distinct books to return (default 20, max 100)"] = 20,
) -> list[dict]:
    """List books in the Alexandria collection, optionally filtered by author, subject or language.

    Returns unique books (one entry per Archive.org identifier) with metadata.
    At least one filter parameter is recommended — without filters, results are arbitrary.

    Args:
        author:   Filter by author/creator name, e.g. 'Kant', 'Nietzsche', 'Plato'.
                  Case-insensitive substring match against the creator field.
        subject:  Filter by subject keyword, e.g. 'ethics', 'logic', 'metaphysics'.
                  Case-insensitive substring match against the subject field.
        language: Filter by language code, e.g. 'eng', 'ger', 'lat', 'fre', 'gre', 'rus'.
        limit:    Maximum number of distinct books to return (default 20, max 100).

    Returns:
        List of books with title, creator, date, language, subject, identifier and total_chunks.
    """
    limit = min(limit, 100)

    conditions = []
    if author:
        conditions.append(FieldCondition(key="creator", match=MatchText(text=author)))
    if subject:
        conditions.append(FieldCondition(key="subject", match=MatchText(text=subject)))
    if language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))

    qfilter = Filter(must=conditions) if conditions else None

    seen: set[str] = set()
    books: list[dict] = []
    offset = None

    while len(books) < limit:
        batch, offset = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qfilter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            identifier = point.payload.get("identifier")
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            p = point.payload
            subj = p.get("subject", "")
            if isinstance(subj, list):
                subj = ", ".join(subj[:3])
            books.append({
                "title":        p.get("title"),
                "creator":      p.get("creator"),
                "date":         p.get("date", "")[:4] if p.get("date") else None,
                "language":     p.get("language"),
                "subject":      subj,
                "identifier":   identifier,
                "total_chunks": p.get("total_chunks"),
            })
            if len(books) >= limit:
                break
        if offset is None:
            break

    _log.info(f'get_book_list author="{author}" subject="{subject}" language="{language}" results={len(books)}')
    return books


@mcp.prompt()
def philosopher_analysis(philosopher: str = "Kant") -> str:
    """Generate a prompt for deep analysis of a philosopher's key ideas."""
    return (
        f"Use search_texts to find primary source passages from {philosopher}. "
        f"Search for their core concepts, major works, and central arguments. "
        f"Synthesise the key themes, examine internal tensions or developments in their thought, "
        f"and place them in historical context."
    )


@mcp.prompt()
def topic_exploration(topic: str = "free will") -> str:
    """Generate a prompt for exploring a philosophical topic across multiple thinkers."""
    return (
        f"Use search_texts to find how different philosophers address '{topic}'. "
        f"Search for several related queries to gather perspectives from multiple thinkers "
        f"and traditions. Compare and contrast their positions, identify points of agreement "
        f"and disagreement, and trace how the debate evolved historically."
    )


@mcp.prompt()
def compare_philosophers(philosopher_a: str = "Plato", philosopher_b: str = "Aristotle", topic: str = "virtue") -> str:
    """Generate a prompt for comparing two philosophers on a specific topic."""
    return (
        f"Use search_texts to find passages from both {philosopher_a} and {philosopher_b} "
        f"on the topic of '{topic}'. Search for each thinker separately, then compare: "
        f"Where do they agree? Where do they diverge fundamentally? "
        f"What are the implications of their different approaches?"
    )


class AcceptPatchMiddleware:
    """Ensure Accept header includes both content types required by streamable-http."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = list(scope["headers"])
            accept_idx = next((i for i, (k, _) in enumerate(headers) if k == b"accept"), None)
            current = headers[accept_idx][1].decode() if accept_idx is not None else ""
            if "text/event-stream" not in current:
                new_accept = (current + ", application/json, text/event-stream").lstrip(", ")
                if accept_idx is not None:
                    headers[accept_idx] = (b"accept", new_accept.encode())
                else:
                    headers.append((b"accept", new_accept.encode()))
                scope["headers"] = headers
        await self.app(scope, receive, send)


if __name__ == "__main__":
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    port = int(os.getenv("MCP_PORT", 8005))
    print(f"→ Starting Alexandria MCP server at http://0.0.0.0:{port}/mcp")
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
        stateless_http=True,
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
            Middleware(AcceptPatchMiddleware),
        ],
    )
