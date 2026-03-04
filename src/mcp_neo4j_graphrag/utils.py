import argparse
import logging
import os
import re
from typing import Any, Union

import tiktoken

logger = logging.getLogger("mcp_neo4j_graphrag")
logger.setLevel(logging.INFO)

# Layer 3: Size-based filtering
MAX_LIST_SIZE = 128        # Drop lists larger than this
MAX_STRING_SIZE = 10000    # Truncate strings longer than this (chars)

# Layer 4: Token-aware truncation - Maximum tokens in tool response
RESPONSE_TOKEN_LIMIT = 8000


def parse_boolean_safely(value: Union[str, bool]) -> bool:
    """Safely parse a string value to boolean with strict validation."""
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        elif normalized == "false":
            return False
        else:
            raise ValueError(
                f"Invalid boolean value: '{value}'. Must be 'true' or 'false'"
            )
    else:
        raise ValueError(f"Invalid boolean value: '{value}'. Must be 'true' or 'false'")


def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|INSERT|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )


def _value_sanitize(
    d: Any, 
    list_limit: int = MAX_LIST_SIZE,
    string_limit: int = MAX_STRING_SIZE
) -> Any:
    """
    Sanitize the input dictionary or list by filtering out large values.
    - Lists with >= list_limit elements are replaced with descriptive placeholders
    - Strings with >= string_limit chars are truncated with suffix
    This prevents embeddings, large text, and binary data from overwhelming the LLM context.
    """
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                new_dict[key] = _value_sanitize(value, list_limit, string_limit)
            elif isinstance(value, list):
                if len(value) >= list_limit:
                    new_dict[key] = f"<list with {len(value)} items (truncated, limit: {list_limit})>"
                else:
                    new_dict[key] = _value_sanitize(value, list_limit, string_limit)
            elif isinstance(value, str):
                if len(value) >= string_limit:
                    new_dict[key] = value[:string_limit] + f"... <truncated at {string_limit} chars, total: {len(value)}>"
                else:
                    new_dict[key] = value
            elif hasattr(value, '__len__') and not isinstance(value, (bytes,)):
                # Catches Neo4j VECTOR type or other sequence-like objects not covered above
                if len(value) >= list_limit:
                    new_dict[key] = f"<vector/sequence with {len(value)} items (truncated)>"
                else:
                    new_dict[key] = list(value)
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(d, list):
        if len(d) >= list_limit:
            return f"<list with {len(d)} items (truncated, limit: {list_limit})>"
        else:
            return [_value_sanitize(item, list_limit, string_limit) for item in d]
    elif isinstance(d, str):
        if len(d) >= string_limit:
            return d[:string_limit] + f"... <truncated at {string_limit} chars, total: {len(d)}>"
        else:
            return d
    elif hasattr(d, '__len__') and not isinstance(d, (bytes,)):
        # Top-level Neo4j VECTOR type or other sequence
        if len(d) >= list_limit:
            return f"<vector/sequence with {len(d)} items (truncated)>"
        else:
            return list(d)
    else:
        return d


def _truncate_string_to_tokens(
    text: str, token_limit: int, model: str = "gpt-4"
) -> str:
    """Truncates the input string to fit within the specified token limit."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]
    truncated_text = encoding.decode(tokens)
    return truncated_text


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def _truncate_results_to_token_limit(
    results: list[dict[str, Any]], 
    token_limit: int, 
    model: str = "gpt-4"
) -> tuple[list[dict[str, Any]], bool]:
    """
    Truncate results by dropping from the end until under token limit.
    Returns (truncated_results, was_truncated).
    """
    import json
    
    # Try with all results first
    result_json = json.dumps(results, default=str)
    token_count = _count_tokens(result_json, model)
    
    if token_count <= token_limit:
        return results, False
    
    # Drop results from the end until under limit
    truncated_results = results.copy()
    while len(truncated_results) > 0:
        truncated_results.pop()
        result_json = json.dumps(truncated_results, default=str)
        token_count = _count_tokens(result_json, model)
        if token_count <= token_limit:
            logger.warning(
                f"Results truncated from {len(results)} to {len(truncated_results)} "
                f"items to stay under {token_limit} token limit"
            )
            return truncated_results, True
    
    # If even empty results exceed limit, return empty
    logger.warning(f"All results dropped - even metadata exceeds {token_limit} token limit")
    return [], True


def process_config(args: argparse.Namespace) -> dict[str, Union[str, int, None]]:
    """
    Process command line arguments and environment variables to create config dictionary.
    """
    config = dict()

    # Parse Neo4j URI
    if args.db_url is not None:
        config["db_url"] = args.db_url
    else:
        config["db_url"] = (
            os.getenv("NEO4J_URL")
            or os.getenv("NEO4J_URI")
            or "bolt://localhost:7687"
        )
        if config["db_url"] == "bolt://localhost:7687":
            logger.warning("Warning: Using default Neo4j URL: bolt://localhost:7687")

    # Parse username
    if args.username is not None:
        config["username"] = args.username
    else:
        config["username"] = os.getenv("NEO4J_USERNAME") or "neo4j"
        if config["username"] == "neo4j":
            logger.warning("Warning: Using default Neo4j username: neo4j")

    # Parse password
    if args.password is not None:
        config["password"] = args.password
    else:
        config["password"] = os.getenv("NEO4J_PASSWORD") or "password"
        if config["password"] == "password":
            logger.warning("Warning: Using default Neo4j password: password")

    # Parse database
    if args.database is not None:
        config["database"] = args.database
    else:
        config["database"] = os.getenv("NEO4J_DATABASE") or "neo4j"

    # Parse embedding model (LiteLLM supports multiple providers)
    if args.embedding_model is not None:
        config["embedding_model"] = args.embedding_model
    else:
        config["embedding_model"] = os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL")
        if not config["embedding_model"]:
            config["embedding_model"] = "text-embedding-3-small"
            logger.warning(
                "EMBEDDING_MODEL not set. Defaulting to 'text-embedding-3-small' (OpenAI). "
                "Set EMBEDDING_MODEL env var to use other providers (e.g., 'azure/...', 'bedrock/...', 'cohere/...')."
            )

    # Parse namespace
    if args.namespace is not None:
        config["namespace"] = args.namespace
    else:
        config["namespace"] = os.getenv("NEO4J_NAMESPACE") or ""

    # Parse transport
    if args.transport is not None:
        config["transport"] = args.transport
    else:
        config["transport"] = os.getenv("NEO4J_TRANSPORT") or "stdio"

    # Parse server host
    if args.server_host is not None:
        config["host"] = args.server_host
    else:
        config["host"] = os.getenv("NEO4J_MCP_SERVER_HOST") or "127.0.0.1"

    # Parse server port
    if args.server_port is not None:
        config["port"] = args.server_port
    else:
        port_env = os.getenv("NEO4J_MCP_SERVER_PORT")
        config["port"] = int(port_env) if port_env else 8000

    # Parse server path
    if args.server_path is not None:
        config["path"] = args.server_path
    else:
        config["path"] = os.getenv("NEO4J_MCP_SERVER_PATH") or "/mcp/"

    # Parse read timeout
    if args.read_timeout is not None:
        config["read_timeout"] = args.read_timeout
    else:
        timeout_env = os.getenv("NEO4J_READ_TIMEOUT")
        config["read_timeout"] = int(timeout_env) if timeout_env else 30

    # Parse schema sample size
    if args.schema_sample_size is not None:
        config["schema_sample_size"] = args.schema_sample_size
    else:
        sample_env = os.getenv("NEO4J_SCHEMA_SAMPLE_SIZE")
        config["schema_sample_size"] = int(sample_env) if sample_env else 1000

    return config

