# Changelog

## [0.4.0] - 2026-03-04

### Added
- **`write_neo4j_cypher`** — execute write Cypher queries (CREATE, MERGE, SET, DELETE, etc.) and return change counters (nodes created, properties set, etc.)
- **`read_node_image`** — retrieve a base64-encoded image stored on a Neo4j node and return it as an inline image; enables multimodal visual analysis of graph-stored content (page scans, diagrams, photos)
- **`pre_filter` parameter on `vector_search`** — filter vector search results by exact property match (e.g. `{"documentName": "report.pdf"}`) compatible with all Neo4j versions

### Fixed
- `write_neo4j_cypher`: removed incorrect `result_transformer_` argument that caused an `AsyncResult` await error
- `vector_search` with `pre_filter`: replaced Neo4j 2025.01+-only 4-argument `queryNodes` syntax with a WHERE clause approach that works on all versions

### Docs
- README: added `write_neo4j_cypher` and `read_node_image` tool documentation; clarified demo database usage for examples; noted that `read_node_image` requires a database storing images on nodes
- `docs/ADVANCED.md`: updated feature comparison table; added detailed reference for new tools including `write_neo4j_cypher`, `read_node_image` workflow, and `pre_filter` usage

---

## [0.3.0] - 2026-01-13

### Added
- Initial release: unified Neo4j GraphRAG MCP server
- `get_neo4j_schema_and_indexes` — schema discovery with vector/fulltext index listing and property size warnings
- `vector_search` — semantic similarity search via embeddings
- `fulltext_search` — keyword search with Lucene syntax
- `read_neo4j_cypher` — read-only Cypher query execution
- `search_cypher_query` — combined vector/fulltext search with graph traversal in a single Cypher query
- Multi-provider embedding support via LiteLLM (OpenAI, Azure, Bedrock, Cohere, Ollama)
- Output sanitization: large list/string truncation + token-aware 8k-token limit
