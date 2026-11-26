import argparse
import asyncio

from . import server
from .utils import process_config


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="Neo4j GraphRAG MCP Server")
    parser.add_argument("--db-url", default=None, help="Neo4j connection URL")
    parser.add_argument("--username", default=None, help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password")
    parser.add_argument("--database", default=None, help="Neo4j database name")
    parser.add_argument(
        "--embedding-model", 
        default=None, 
        help="Embedding model (default: text-embedding-3-small). Supports OpenAI, Azure, Bedrock, Cohere, etc. via LiteLLM."
    )
    parser.add_argument(
        "--transport", default=None, help="Transport type (stdio, sse, http)"
    )
    parser.add_argument("--namespace", default=None, help="Tool namespace")
    parser.add_argument(
        "--server-path", default=None, help="HTTP path (default: /mcp/)"
    )
    parser.add_argument("--server-host", default=None, help="Server host")
    parser.add_argument("--server-port", type=int, default=None, help="Server port")
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for read queries (default: 30)",
    )
    parser.add_argument(
        "--schema-sample-size",
        type=int,
        default=None,
        help="Default sample size for schema operations (default: 1000)",
    )

    parser.add_argument("--ontology-file", default=None, help="Ontology Graph Spec YAML/JSON file for get_ontology (or ONTOLOGY_FILE)")
    parser.add_argument("--ontology-name", default=None, help="Default ontology name in the template Ontology DB (or ONTOLOGY_NAME)")
    args = parser.parse_args()
    config = process_config(args)
    asyncio.run(server.main(**config))


__all__ = ["main", "server"]

