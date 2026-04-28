# Alexandria MCP Server

Semantic search over **4.6 million text chunks** from 20,000+ classical philosophy and humanities works. Built for AI agents using [FastMCP](https://github.com/jlowin/fastmcp) over HTTP.

## What's in the collection

All texts are public domain (pre-1928), sourced from [Internet Archive](https://archive.org) (americana, europeanlibraries, gutenberg collections).

- **Philosophy:** Aristotle, Plato, Kant, Hegel, Nietzsche, Schopenhauer, Descartes, Spinoza, Locke, Hume, Mill, Wittgenstein, Aquinas, Augustine, Leibniz, Rousseau, Voltaire, Marx, and hundreds more
- **Topics:** Ethics, metaphysics, epistemology, logic, political philosophy, theology, stoicism, neoplatonism, existentialism, history of ideas
- **Languages:** English, German, Latin, French, Italian, Greek, Russian

## Connecting to the server

**Remote (hosted):**
```bash
claude mcp add --transport http alexandria https://alexandria.aidatanorge.no/mcp
```

**Via MCP config:**
```json
{
  "mcpServers": {
    "alexandria": {
      "type": "http",
      "url": "https://alexandria.aidatanorge.no/mcp"
    }
  }
}
```

## Tools

### `search_texts`
Search the collection using natural language. Uses hybrid dense+sparse retrieval with cross-encoder reranking.

| Parameter | Type | Description |
|---|---|---|
| `query` | string | What you are looking for |
| `language` | string | Optional language filter: `eng`, `ger`, `lat`, `fre`, `ita`, `gre`, `rus` |
| `limit` | int | Number of results (default 5, max 20) |

**Example queries:**
- `"Nietzsche will to power eternal recurrence"`
- `"Kantian categorical imperative duty"`
- `"Platonic theory of forms and the Good"`
- `"Stoic virtue and the sage"`
- `"Hegel dialectics spirit history"`

Each result includes: `title`, `creator`, `date`, `language`, `subject`, `text` (chunk), `rerank_score`, `vector_score`.

### `ping`
Connectivity test.

## Prompts

- `philosopher_analysis(philosopher)` — deep dive into a philosopher's key ideas
- `topic_exploration(topic)` — explore a topic across multiple thinkers
- `compare_philosophers(philosopher_a, philosopher_b, topic)` — compare two philosophers on a specific topic

## Architecture

```
Archive.org (13,000+ books)
        ↓
  Text extraction + chunking
        ↓
  Qdrant (4.6M vectors)
        ↓
  Hybrid search: intfloat/multilingual-e5-large (dense) + Qdrant/bm25 (sparse)
        ↓
  Cross-encoder reranking: mmarco-mMiniLMv2-L12-H384-v1
        ↓
  FastMCP 3.2 over HTTP
```

## Self-hosting

Requires a running Qdrant instance with the `alexandria` collection populated.

```bash
pip install -r requirements.txt
python alexandria_mcp_server.py
# Server starts at http://localhost:8005/mcp
```

Environment variables (optional):
```
QDRANT_HOST=localhost
QDRANT_PORT=6333
MCP_PORT=8005
```

## License

MIT
