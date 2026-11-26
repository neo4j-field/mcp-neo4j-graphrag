# Neo4j GraphRAG MCP Server

[![PyPI version](https://badge.fury.io/py/mcp-neo4j-graphrag.svg)](https://pypi.org/project/mcp-neo4j-graphrag/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An MCP server that extends Neo4j with **vector search**, **fulltext search**, **search-augmented Cypher queries**, **write operations**, and **multimodal image retrieval** for GraphRAG applications.

> **Inspired by** the [Neo4j Labs `mcp-neo4j-cypher`](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher) server. This server adds vector search, fulltext search, and the innovative `search_cypher_query` tool for combining search with graph traversal.

## Overview

This server enables LLMs to:
- 🔍 Search Neo4j vector indexes using semantic similarity
- 📝 Search fulltext indexes with Lucene syntax
- ⚡ Combine search with Cypher queries via `search_cypher_query`
- 🕸️ Execute read-only Cypher queries
- ✏️ Execute write Cypher queries (CREATE, MERGE, SET, DELETE)
- 🖼️ Retrieve images stored in Neo4j nodes (multimodal — returns the image directly to the LLM)
- Navigate lexical documents by section hierarchy, then read complete source elements with page citations
- Read domain meaning from an Ontology Graph Spec file or the template Ontology DB

Built on [LiteLLM](https://docs.litellm.ai/) for multi-provider embedding support (OpenAI, Azure, Bedrock, Cohere, etc.).

> **Related:** For the official Neo4j MCP Server, see [neo4j/mcp](https://github.com/neo4j/mcp). For Neo4j Labs MCP Servers (Cypher, Memory, Data Modeling), see [neo4j-contrib/mcp-neo4j](https://github.com/neo4j-contrib/mcp-neo4j).

## Installation

```bash
# Using pip
pip install mcp-neo4j-graphrag

# Using uv (recommended)
uv pip install mcp-neo4j-graphrag
```

## Configuration

### Claude Desktop

Edit the configuration file:
- **macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "neo4j-graphrag": {
      "command": "uvx",
      "args": ["mcp-neo4j-graphrag"],
      "env": {
        "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com",
        "NEO4J_USERNAME": "recommendations",
        "NEO4J_PASSWORD": "recommendations",
        "NEO4J_DATABASE": "recommendations",
        "OPENAI_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    }
  }
}
```

> **Note**: `uvx` automatically downloads and runs the package from PyPI. No local installation needed!

### Cursor

Edit `~/.cursor/mcp.json` or `.cursor/mcp.json` in your project. Use the same configuration as above.

### Reload Configuration

- **Claude Desktop:** Quit and restart the application
- **Cursor:** Reload the window (Cmd/Ctrl + Shift + P → "Reload Window")

## Tools

The examples below use the [Neo4j demo `recommendations` database](https://demo.neo4jlabs.com) (movies, actors, directors), which is the same database referenced in the Configuration section above.

### Document-tree retrieval (PageIndex-style)

These read-only tools navigate the workspace lexical graph without embeddings,
APOC, summaries or model calls. They implement the outline → choose → read pattern
from the supplied PageIndex-Graph workshop reference.
The server returns evidence; the calling agent reasons about which sections to read.

| Tool | Purpose |
|------|---------|
| `list_retrieval_documents(query, offset, limit)` | Discover active document versions; names are searchable, IDs select precisely |
| `get_document_outline(document_id, root_section_id, depth, offset, limit)` | Browse titles, ancestor paths, page ranges and child availability; drill using section IDs |
| `get_section_content(document_id, section_id, include_descendants, ...)` | Read original section elements, with explicit source/page citations |
| `get_document_content(document_id, ...)` | Read front matter, flat PDFs or a whole document incrementally |

Document tools are advertised only when the connected database contains usable
active lexical content. Outline/section tools additionally require section-to-element
structure. A business-only or empty graph exposes none of these navigation tools.
Availability is checked again whenever tools are listed and when a navigation tool
is called. After ingesting or removing data, refresh the agent's tool list or
reconnect if its client caches discovery results. Other GraphRAG tools remain available.

Start with a document outline, choose a section, then read its content. Pass the
returned `next.offset` and `next.text_offset` to continue; `next_offset` pages an
outline or document list. A long element can span multiple calls without losing
its tail. `has_more=false` ends that view; it does not prove a fact absent from the
entire corpus. Content is scoped to an active exact document version; old chunk
sets are excluded. Fetch selected visual evidence using `read_node_image` with a
returned `node_element_id`.

The native structure uses `Document-HAS_SECTION->Section`,
`Section-HAS_SUBSECTION->Section` and shared `Element` membership. Document links
include **all** sections, not just roots. Element/page/chunk fallback is explicit
for parsers without sections. Citations expose 1-based `page_number` alongside
stored 0-based `page_index`; unavailable page numbers remain null. Pageless
Markdown, TXT, local HTML and DOCX return `source_format` in document metadata and
citations, plus each element's stored `source_anchor` (for example `element:3` or
`lines:4-6`) on the item and citation. Their page fields remain null; use the
source path, anchor and native section path rather than an invented page number.
`node_element_id` remains available for reading linked visual evidence. Newly parsed
sections have order and descendant page ranges; legacy missing ranges remain null.
No parser migration or new index is required. Pagination assumes the selected
version is not modified between calls. These tools do not add cross-document links,
section summaries, themes or the upstream PageIndex algorithm.

### `get_ontology`

Use domain semantics before writing business Cypher. Inspired by the
[text2cypher ontology agent](https://github.com/neo4j-field/text2cypher-ontology-agent),
`get_ontology(entities=["Film"])` resolves IDs, database names and semantic aliases;
`get_ontology(search="rating")` finds relevant vocabulary and descriptions. Selected
relationships include their endpoint nodes. A compact schema card accompanies the
structured ontology; oversized responses become a bounded entity index for drilldown.

Configure the source at server startup:

```bash
# Read an existing Neo4j Ontology Graph Spec 1.1.0 YAML/JSON artifact:
ONTOLOGY_FILE=/absolute/path/domain.ontology.yaml uv run mcp-neo4j-graphrag
# Equivalent CLI flag: --ontology-file /absolute/path/domain.ontology.yaml
```

Without a configured file, the tool reads the template's existing Ontology DB.
Set `NEO4J_ONTOLOGY_DATABASE` and optionally `NEO4J_ONTOLOGY_URI`,
`NEO4J_ONTOLOGY_USERNAME`, `NEO4J_ONTOLOGY_PASSWORD` for a separate instance.
`ONTOLOGY_NAME` / `--ontology-name` selects a default; the tool accepts
`ontology_name` and lists available names when selection is ambiguous.

File mode validates the **full pinned specification** offline, retaining aliases,
cardinalities, value constraints and extensions; it is broader than the extraction
adapter's supported subset. DB mode projects the template's private ontology model
and normalizer extensions. No future DBMS ontology procedure is assumed. The agent
cannot choose arbitrary filesystem paths. Ontology meaning is declared intent, not
proof that data or indexes exist: use physical schema inspection for gaps/indexes
and read queries for actual values or counts. The ontology does not enforce queries.

### `get_neo4j_schema_and_indexes`

Discover the graph schema, vector indexes, and fulltext indexes.

💡 Use this for physical schema/index discovery before index-based search, or when the ontology lacks the needed domain. Document-tree navigation requires no schema-discovery call.

**Example prompt:**
> "What is inside the database?"

### `vector_search`

Semantic similarity search using embeddings.

**Parameters:** `text_query`, `vector_index`, `top_k`, `return_properties`, `pre_filter`

Use `pre_filter` to restrict results to nodes matching exact property values (e.g. `{"genre": "Drama"}`).

**Example prompt:**
> "What movies are about artificial intelligence?"

### `fulltext_search`

Keyword search with Lucene syntax (AND, OR, wildcards, fuzzy).

**Parameters:** `text_query`, `fulltext_index`, `top_k`, `return_properties`

**Example prompt:**
> "Find movies with 'space' or 'galaxy' in the title or plot"

### `read_neo4j_cypher`

Execute read-only Cypher queries.

**Parameters:** `query`, `params`

**Example prompt:**
> "Show me all genres and how many movies are in each"

### `search_cypher_query`

Combine vector/fulltext search with Cypher queries. Use `$vector_embedding` and `$fulltext_text` placeholders.

**Parameters:** `cypher_query`, `vector_query`, `fulltext_query`, `params`

**Example prompt:**
> "In one query, what are the directors and genres of the movies about 'time travel adventure'?"

### `write_neo4j_cypher`

Execute write Cypher queries (CREATE, MERGE, SET, DELETE, etc.). Returns a summary of counters (nodes created, properties set, etc.).

**Parameters:** `query`, `params`

**Example prompt:**
> "Add a user rating of 4.5 for the movie 'Inception'"

### `read_node_image`

Retrieve a base64-encoded image stored on a Neo4j node and return it as an inline image. Useful for graph databases that store page scans, diagrams, or photos directly on nodes. The LLM receives both the image and selected node properties, enabling visual analysis of graph-stored content.

**Parameters:** `node_element_id`, `image_property`, `mime_type`, `return_properties`

> **Note:** This tool requires a database that stores images directly on nodes (as base64). The demo `recommendations` database does not — it stores external poster URLs instead. See [docs/ADVANCED.md](docs/ADVANCED.md) for a full example using a document graph where page images are embedded on nodes.

**Example prompt:**
> "Show me page 3 of the AbbVie pipeline document and describe what you see"

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | `password` | Neo4j password |
| `NEO4J_DATABASE` | No | `neo4j` | Database name |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model (see below) |

### Embedding Providers

Set `EMBEDDING_MODEL` and the corresponding API key:

| Provider | Model Format | API Key Variable |
|----------|-------------|------------------|
| OpenAI | `text-embedding-ada-002` | `OPENAI_API_KEY` |
| Azure | `azure/deployment-name` | `AZURE_API_KEY`, `AZURE_API_BASE` |
| Bedrock | `bedrock/amazon.titan-embed-text-v1` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Cohere | `cohere/embed-english-v3.0` | `COHERE_API_KEY` |
| Ollama | `ollama/nomic-embed-text` | *(none - local)* |

## Advanced Topics

See [docs/ADVANCED.md](docs/ADVANCED.md) for:
- Comparison with Neo4j Labs `mcp-neo4j-cypher` server
- Production features (output sanitization, token limits)
- Detailed tool documentation including `write_neo4j_cypher`, `read_node_image`, and `vector_search` filtering

## License

MIT License
