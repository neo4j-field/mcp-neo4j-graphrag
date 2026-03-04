import json
import logging
from typing import Any, Literal, Optional

import litellm
from dotenv import load_dotenv
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ImageContent, ToolAnnotations
from neo4j import AsyncDriver, AsyncGraphDatabase, Query, RoutingControl
from neo4j.exceptions import ClientError, Neo4jError
from pydantic import Field

from .utils import (
    _is_write_query,
    _value_sanitize,
    _truncate_string_to_tokens,
    _truncate_results_to_token_limit,
    MAX_LIST_SIZE,
    MAX_STRING_SIZE,
    RESPONSE_TOKEN_LIMIT,
)

load_dotenv()

logger = logging.getLogger("mcp_neo4j_graphrag")


def _format_namespace(namespace: str) -> str:
    """Format namespace with trailing dash if needed."""
    if namespace:
        return namespace if namespace.endswith("-") else namespace + "-"
    return ""


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    embedding_model: str,
    database: str = "neo4j",
    namespace: str = "",
    read_timeout: int = 30,
    config_sample_size: int = 1000,
) -> FastMCP:
    """Create the unified GraphRAG MCP server."""
    
    mcp: FastMCP = FastMCP("mcp-neo4j-graphrag")
    namespace_prefix = _format_namespace(namespace)

    # ========================================
    # DISCOVERY TOOL
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "get_neo4j_schema_and_indexes",
        annotations=ToolAnnotations(
            title="Get Neo4j Schema & Indexes",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_neo4j_schema_and_indexes(
        sample_size: int = Field(
            default=config_sample_size,
            description="The sample size used to infer the graph schema and property sizes. Larger samples are slower but more accurate."
        )
    ) -> list[ToolResult]:
        """
        Returns Neo4j graph schema with search indexes and property size warnings.
        
        **IMPORTANT: Call this tool BEFORE using any search tools (vector_search, fulltext_search, search_cypher_query).**
        
        This tool provides:
        - Vector & fulltext indexes (for search)
        - Node/relationship schemas with property types
        - Warnings for large properties (helps choose efficient return_properties)
        
        Property size warnings help you avoid token limits when using search tools.
        For example, if a property has warning "avg ~100-200KB", avoid returning it unless necessary.
        
        You should only provide a `sample_size` value if requested by the user, or tuning performance.
        """
        effective_sample_size = sample_size if sample_size else config_sample_size
        logger.info(f"Running `get_neo4j_schema_and_indexes` with sample size {effective_sample_size}")

        # Step 1: Get search indexes
        vector_index_query = """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, options
        WHERE type = 'VECTOR'
        RETURN name, entityType, labelsOrTypes, properties, options
        """

        fulltext_index_query = """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, options
        WHERE type = 'FULLTEXT'
        RETURN name, entityType, labelsOrTypes, properties, options
        """

        # Step 2: Get schema using APOC
        get_schema_query = f"CALL apoc.meta.schema({{sample: {effective_sample_size}}}) YIELD value RETURN value"

        try:
            # Fetch indexes
            vector_indexes = await neo4j_driver.execute_query(
                vector_index_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            fulltext_indexes = await neo4j_driver.execute_query(
                fulltext_index_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            # Fetch schema
            schema_results = await neo4j_driver.execute_query(
                get_schema_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            # Step 3: Sample property sizes for indexed labels
            indexed_labels = set()
            for idx in vector_indexes:
                indexed_labels.update(idx.get("labelsOrTypes", []))
            for idx in fulltext_indexes:
                indexed_labels.update(idx.get("labelsOrTypes", []))

            property_size_warnings = {}
            for label in indexed_labels:
                # Sample property sizes for this label
                size_query = f"""
                MATCH (n:{label})
                WITH n LIMIT {min(effective_sample_size, 100)}
                WITH n, properties(n) as props
                UNWIND keys(props) as propName
                WITH propName, props[propName] as propValue
                WHERE propValue IS NOT NULL
                WITH propName,
                     valueType(propValue) as propType,
                     CASE
                         WHEN valueType(propValue) STARTS WITH 'LIST' THEN size(propValue)
                         WHEN valueType(propValue) STARTS WITH 'STRING' THEN size(propValue)
                         ELSE 0
                     END as propSize
                RETURN propName, propType, avg(propSize) as avgSize, max(propSize) as maxSize
                ORDER BY avgSize DESC
                """
                
                try:
                    size_results = await neo4j_driver.execute_query(
                        size_query,
                        routing_control=RoutingControl.READ,
                        database_=database,
                        result_transformer_=lambda r: r.data(),
                    )
                    
                    if label not in property_size_warnings:
                        property_size_warnings[label] = {}
                    
                    logger.debug(f"Sampled {len(size_results)} properties for label {label}")
                    
                    for row in size_results:
                        prop_name = row["propName"]
                        prop_type = row["propType"]
                        avg_size = row["avgSize"]
                        max_size = row["maxSize"]
                        
                        logger.debug(f"  {prop_name} ({prop_type}): avg={avg_size}, max={max_size}")
                        
                        # Generate warning for large properties based on type
                        warning = None
                        if prop_type.startswith("STRING"):
                            # For strings, warn if >= 100KB
                            if avg_size >= 100000:
                                warning = f"very large (avg ~{int(avg_size/1000)}KB, max ~{int(max_size/1000)}KB)"
                        elif prop_type.startswith("LIST"):
                            # For lists, warn if >= 1000 items (likely embeddings)
                            if avg_size >= 1000:
                                warning = f"large list (avg ~{int(avg_size)} items)"
                        
                        if warning:
                            property_size_warnings[label][prop_name] = warning
                
                except Exception as e:
                    logger.warning(f"Could not sample property sizes for label {label}: {e}")
                    continue

            # Step 4: Clean and enrich schema with warnings
            def clean_and_enrich_schema(schema: dict) -> dict:
                cleaned = {}
                for key, entry in schema.items():
                    new_entry = {"type": entry["type"]}
                    if "count" in entry:
                        new_entry["count"] = entry["count"]

                    labels = entry.get("labels", [])
                    if labels:
                        new_entry["labels"] = labels

                    props = entry.get("properties", {})
                    clean_props = {}
                    for pname, pinfo in props.items():
                        cp = {}
                        if "indexed" in pinfo:
                            cp["indexed"] = pinfo["indexed"]
                        if "type" in pinfo:
                            cp["type"] = pinfo["type"]
                        
                        # Add size warning if available
                        if key in property_size_warnings and pname in property_size_warnings[key]:
                            cp["warning"] = property_size_warnings[key][pname]
                        
                        if cp:
                            clean_props[pname] = cp
                    if clean_props:
                        new_entry["properties"] = clean_props

                    if entry.get("relationships"):
                        rels_out = {}
                        for rel_name, rel in entry["relationships"].items():
                            cr = {}
                            if "direction" in rel:
                                cr["direction"] = rel["direction"]
                            rlabels = rel.get("labels", [])
                            if rlabels:
                                cr["labels"] = rlabels
                            rprops = rel.get("properties", {})
                            clean_rprops = {}
                            for rpname, rpinfo in rprops.items():
                                crp = {}
                                if "indexed" in rpinfo:
                                    crp["indexed"] = rpinfo["indexed"]
                                if "type" in rpinfo:
                                    crp["type"] = rpinfo["type"]
                                if crp:
                                    clean_rprops[rpname] = crp
                            if clean_rprops:
                                cr["properties"] = clean_rprops
                            if cr:
                                rels_out[rel_name] = cr
                        if rels_out:
                            new_entry["relationships"] = rels_out

                    cleaned[key] = new_entry
                return cleaned

            schema_clean = clean_and_enrich_schema(schema_results[0].get("value"))

            # Step 5: Combine everything into compact JSON
            result = {
                "indexes": {
                    "vector": vector_indexes,
                    "fulltext": fulltext_indexes
                },
                "schema": schema_clean
            }

            result_json = json.dumps(result, default=str)
            logger.debug(f"Found {len(vector_indexes)} vector indexes, {len(fulltext_indexes)} fulltext indexes")

            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except ClientError as e:
            if "Neo.ClientError.Procedure.ProcedureNotFound" in str(e):
                raise ToolError(
                    "Neo4j Client Error: This instance of Neo4j does not have the APOC plugin installed. Please install and enable APOC."
                )
            else:
                raise ToolError(f"Neo4j Client Error: {e}")

        except Neo4jError as e:
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error retrieving Neo4j schema and indexes: {e}")
            raise ToolError(f"Unexpected Error: {e}")

    # ========================================
    # SIMPLE SEARCH TOOLS
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "vector_search",
        annotations=ToolAnnotations(
            title="Vector Similarity Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def vector_search(
        text_query: str = Field(..., description="The text query to search for. This will be embedded and used for similarity search."),
        vector_index: str = Field(..., description="The name of the vector index to search in. Use get_neo4j_schema_and_indexes to see available indexes."),
        top_k: int = Field(default=5, description="The number of most similar results to return."),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of properties to return (e.g., "pageNumber,id"). If not specified, returns all properties with automatic sanitization (large values are truncated).'
        ),
        pre_filter: Optional[dict] = Field(
            None,
            description='Optional: Filter map applied after vector search (e.g., {"documentName": "foo.pdf"}). Filters results by exact property match. Check get_neo4j_schema_and_indexes for available node properties.'
        ),
    ) -> list[ToolResult]:
        """
        Performs vector similarity search on a Neo4j vector index.

        This tool embeds your text query using OpenAI and searches the specified vector index.
        Returns node IDs, labels, node properties (automatically sanitized), and similarity scores.

        **Automatic Sanitization (always applied):**
        - Embedding property used by the vector index → automatically excluded (vector_search only)
        - Large lists (≥128 items) → replaced with placeholders
        - Large strings (≥10K chars) → truncated with suffix
        - Total response limited to 8000 tokens (results dropped if needed)

        **Property Selection:**
        - Default (no return_properties): Returns ALL properties (sanitized)
        - With return_properties: Returns ONLY specified properties
        - Example: return_properties="pageNumber,id" → returns only these two
        - Check get_neo4j_schema_and_indexes for property warnings to avoid large fields

        **Post-Filtering:**
        - Use pre_filter to filter results by exact property match after vector scoring (e.g., {"documentName": "foo.pdf"})
        - Check get_neo4j_schema_and_indexes for available node properties to filter on

        **Performance Optimization:**
        Internally fetches max(top_k × 2, 100) results to avoid local maximum problems in kANN algorithms.
        """
        logger.info(f"Running `vector_search` with query='{text_query}', index='{vector_index}', top_k={top_k}, return_properties={return_properties}, pre_filter={pre_filter}")

        try:
            # Get the embedding property name from the vector index
            index_info_query = """
            SHOW INDEXES
            YIELD name, type, properties
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN properties
            """
            
            index_info = await neo4j_driver.execute_query(
                index_info_query,
                parameters_={"index_name": vector_index},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )
            
            if not index_info:
                raise ToolError(f"Vector index '{vector_index}' not found. Use get_neo4j_schema_and_indexes to see available indexes.")
            
            embedding_property = index_info[0]["properties"][0] if index_info[0]["properties"] else None
            logger.debug(f"Embedding property for index '{vector_index}': {embedding_property}")

            # Generate embedding using LiteLLM (supports multiple providers)
            logger.debug(f"Generating embedding with model: {embedding_model}")
            embedding_response = litellm.embedding(
                model=embedding_model,
                input=[text_query]
            )
            query_embedding = embedding_response.data[0]["embedding"]

            # Fetch more results to avoid local maximum
            fetch_k = max(top_k * 2, 100)
            logger.debug(f"Fetching {fetch_k} results from vector index")

            # Parse return_properties if provided (comma-separated string)
            property_list = None
            if return_properties:
                property_list = [prop.strip() for prop in return_properties.split(",")]
                logger.debug(f"Parsed return_properties: {property_list}")

            # Build RETURN clause based on return_properties
            search_params: dict[str, Any] = {
                "index_name": vector_index,
                "fetch_k": fetch_k,
                "top_k": top_k,
                "query_vector": query_embedding,
            }

            call_clause = "CALL db.index.vector.queryNodes($index_name, $fetch_k, $query_vector)"

            where_clause = ""
            if pre_filter:
                conditions = " AND ".join([f"node.{k} = $pf_{k}" for k in pre_filter.keys()])
                where_clause = f"WHERE {conditions}"
                for k, v in pre_filter.items():
                    search_params[f"pf_{k}"] = v

            if property_list:
                props_return = ", ".join([f"node.{prop} as {prop}" for prop in property_list])
                search_query = f"""
                {call_clause}
                YIELD node, score
                {where_clause}
                RETURN elementId(node) as nodeId, labels(node) as labels, {props_return}, score
                ORDER BY score DESC
                LIMIT $top_k
                """
            else:
                search_query = f"""
                {call_clause}
                YIELD node, score
                {where_clause}
                RETURN elementId(node) as nodeId, labels(node) as labels, properties(node) as properties, score
                ORDER BY score DESC
                LIMIT $top_k
                """

            results = await neo4j_driver.execute_query(
                search_query,
                parameters_=search_params,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Vector search returned {len(results)} results")

            # Auto-exclude the embedding property (vector_search only)
            if embedding_property:
                for result in results:
                    if 'properties' in result and embedding_property in result['properties']:
                        del result['properties'][embedding_property]
                        logger.debug(f"Auto-excluded embedding property: {embedding_property}")

            # Layer 3: Sanitize large lists and strings (images, text, etc.)
            for result in results:
                if 'properties' in result:
                    result['properties'] = _value_sanitize(result['properties'], MAX_LIST_SIZE, MAX_STRING_SIZE)

            # Layer 4: Truncate results to stay under token limit
            original_count = len(results)
            results, was_truncated = _truncate_results_to_token_limit(results, RESPONSE_TOKEN_LIMIT)

            formatted_results = {
                "query": text_query,
                "index": vector_index,
                "top_k": top_k,
                "results": results
            }

            if was_truncated:
                formatted_results["warning"] = f"Results truncated from {original_count} to {len(results)} items (token limit: {RESPONSE_TOKEN_LIMIT})"

            result_json = json.dumps(formatted_results, default=str, indent=2)
            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error during vector search: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise ToolError(f"Error: {e}")

    @mcp.tool(
        name=namespace_prefix + "fulltext_search",
        annotations=ToolAnnotations(
            title="Fulltext Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def fulltext_search(
        text_query: str = Field(..., description="The text query to search for. Supports Lucene query syntax (AND, OR, wildcards, fuzzy, etc.)."),
        fulltext_index: str = Field(..., description="The name of the fulltext index to search. Use get_neo4j_schema_and_indexes to see available indexes."),
        top_k: int = Field(default=5, description="The number of most relevant results to return."),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of properties to return (e.g., "pageNumber,id"). If not specified, returns all properties with automatic sanitization (large values are truncated).'
        ),
    ) -> list[ToolResult]:
        """
        Performs fulltext search on a Neo4j fulltext index using Lucene query syntax.
        
        **Lucene Syntax Supported:**
        - Boolean: "legal AND compliance", "privacy OR security"
        - Wildcards: "compli*", "te?t"
        - Fuzzy: "complience~"
        - Phrases: "\"exact phrase\""
        
        **Automatic Sanitization (always applied):**
        - Large lists (≥128 items) → replaced with placeholders
        - Large strings (≥10K chars) → truncated with suffix
        - Total response limited to 8000 tokens (results dropped if needed)
        
        **Property Selection:**
        - Default (no return_properties): Returns ALL properties (sanitized)
        - With return_properties: Returns ONLY specified properties
        - Example: return_properties="pageNumber,id" → returns only these two
        - Check get_neo4j_schema_and_indexes for property warnings to avoid large fields
        
        Returns node/relationship IDs, labels/types, properties (sanitized), and relevance scores.
        """
        logger.info(f"Running `fulltext_search` with query='{text_query}', index='{fulltext_index}', top_k={top_k}, return_properties={return_properties}")

        try:
            # Get index entity type
            index_info_query = """
            SHOW INDEXES
            YIELD name, type, entityType
            WHERE name = $index_name AND type = 'FULLTEXT'
            RETURN entityType
            """
            
            index_info = await neo4j_driver.execute_query(
                index_info_query,
                parameters_={"index_name": fulltext_index},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            if not index_info:
                raise ToolError(f"Fulltext index '{fulltext_index}' not found. Use get_neo4j_schema_and_indexes to see available indexes.")

            entity_type = index_info[0]["entityType"]

            # Parse return_properties if provided (comma-separated string)
            property_list = None
            if return_properties:
                property_list = [prop.strip() for prop in return_properties.split(",")]
                logger.debug(f"Parsed return_properties: {property_list}")

            # Build RETURN clause based on return_properties and entity_type
            if entity_type == "NODE":
                if property_list:
                    props_return = ", ".join([f"node.{prop} as {prop}" for prop in property_list])
                    search_query = f"""
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    RETURN elementId(node) as nodeId, labels(node) as labels, {props_return}, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
                else:
                    search_query = """
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    RETURN elementId(node) as nodeId, labels(node) as labels, properties(node) as properties, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
            else:
                if property_list:
                    props_return = ", ".join([f"relationship.{prop} as {prop}" for prop in property_list])
                    search_query = f"""
                    CALL db.index.fulltext.queryRelationships($index_name, $query)
                    YIELD relationship, score
                    RETURN elementId(relationship) as relationshipId, type(relationship) as type, {props_return}, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
                else:
                    search_query = """
                    CALL db.index.fulltext.queryRelationships($index_name, $query)
                    YIELD relationship, score
                    RETURN elementId(relationship) as relationshipId, type(relationship) as type, properties(relationship) as properties, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """

            results = await neo4j_driver.execute_query(
                search_query,
                parameters_={
                    "index_name": fulltext_index,
                    "query": text_query,
                    "top_k": top_k
                },
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Fulltext search returned {len(results)} results")

            # Layer 3: Sanitize large lists and strings (embeddings, images, text, etc.)
            for result in results:
                if 'properties' in result:
                    result['properties'] = _value_sanitize(result['properties'], MAX_LIST_SIZE, MAX_STRING_SIZE)

            # Layer 4: Truncate results to stay under token limit
            original_count = len(results)
            results, was_truncated = _truncate_results_to_token_limit(results, RESPONSE_TOKEN_LIMIT)

            formatted_results = {
                "query": text_query,
                "index": fulltext_index,
                "entity_type": entity_type,
                "top_k": top_k,
                "results": results
            }

            if was_truncated:
                formatted_results["warning"] = f"Results truncated from {original_count} to {len(results)} items (token limit: {RESPONSE_TOKEN_LIMIT})"

            result_json = json.dumps(formatted_results, default=str, indent=2)
            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error during fulltext search: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error during fulltext search: {e}")
            raise ToolError(f"Error: {e}")

    # ========================================
    # CYPHER QUERY TOOLS
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "read_neo4j_cypher",
        annotations=ToolAnnotations(
            title="Read Neo4j Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_neo4j_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: dict[str, Any] = Field(
            dict(), description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """Execute a read Cypher query on the Neo4j database."""

        if _is_write_query(query):
            raise ValueError("Only MATCH queries are allowed for read_neo4j_cypher")

        try:
            query_obj = Query(query, timeout=float(read_timeout))
            results = await neo4j_driver.execute_query(
                query_obj,
                parameters_=params,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Read query returned {len(results)} rows")

            # Layer 3: Sanitize large lists and strings (embeddings, images, text, etc.)
            sanitized_results = [_value_sanitize(el, MAX_LIST_SIZE, MAX_STRING_SIZE) for el in results]

            # Layer 4: Truncate results to stay under token limit
            original_count = len(sanitized_results)
            sanitized_results, was_truncated = _truncate_results_to_token_limit(
                sanitized_results, RESPONSE_TOKEN_LIMIT
            )

            if was_truncated:
                logger.warning(
                    f"Cypher results truncated from {original_count} to {len(sanitized_results)} rows"
                )

            results_json_str = json.dumps(sanitized_results, default=str)
            return ToolResult(content=[TextContent(type="text", text=results_json_str)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error executing read query: {e}\n{query}\n{params}")
            raise ToolError(f"Neo4j Error: {e}\n{query}\n{params}")

        except Exception as e:
            logger.error(f"Error executing read query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    @mcp.tool(
        name=namespace_prefix + "write_neo4j_cypher",
        annotations=ToolAnnotations(
            title="Write Neo4j Cypher",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def write_neo4j_cypher(
        query: str = Field(..., description="The write Cypher query to execute (CREATE, MERGE, SET, DELETE, etc.)."),
        params: dict[str, Any] = Field(
            dict(), description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """Execute a write Cypher query on the Neo4j database and return the result summary (counters)."""

        try:
            eager_result = await neo4j_driver.execute_query(
                query,
                parameters_=params,
                routing_control=RoutingControl.WRITE,
                database_=database,
            )

            counters = vars(eager_result.summary.counters)
            summary = {
                "counters": counters,
                "result_available_after_ms": eager_result.summary.result_available_after,
            }

            logger.debug(f"Write query executed. Counters: {counters}")
            return ToolResult(content=[TextContent(type="text", text=json.dumps(summary, default=str))])

        except Neo4jError as e:
            logger.error(f"Neo4j Error executing write query: {e}\n{query}\n{params}")
            raise ToolError(f"Neo4j Error: {e}\n{query}\n{params}")

        except Exception as e:
            logger.error(f"Error executing write query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    # ========================================
    # SEARCH + CYPHER TOOL (NEW!)
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "search_cypher_query",
        annotations=ToolAnnotations(
            title="Search-Augmented Cypher Query",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_cypher_query(
        cypher_query: str = Field(..., description="Cypher query using $vector_embedding and/or $fulltext_text placeholders."),
        vector_query: Optional[str] = Field(None, description="Text query to embed for vector search. Use $vector_embedding placeholder in Cypher."),
        fulltext_query: Optional[str] = Field(None, description="Text query for fulltext search. Use $fulltext_text placeholder in Cypher."),
        params: dict[str, Any] = Field(dict(), description="Additional parameters for the Cypher query."),
    ) -> list[ToolResult]:
        """
        Execute a Cypher query that uses vector and/or fulltext search indexes.
        
        This powerful tool allows you to:
        1. Use vector search ($vector_embedding) and/or fulltext search ($fulltext_text) in Cypher
        2. Post-filter large result sets (fetch 100-1000, filter with WHERE)
        3. Combine search with graph traversal
        4. Aggregate over search results
        
        **Example:**
        ```python
        search_cypher_query(
            cypher_query='''
                CALL db.index.vector.queryNodes('chunk_embedding_vector', 500, $vector_embedding)
                YIELD node, score
                WHERE score > 0.75
                MATCH (node)-[:BELONGS_TO]->(d:Document)
                WHERE d.year >= 2020
                RETURN node.chunkId, d.title, score
                ORDER BY score DESC
                LIMIT 20
            ''',
            vector_query="student requirements"
        )
        ```
        
        **Placeholders:**
        - `$vector_embedding`: Replaced with embedding vector
        - `$fulltext_text`: Replaced with text string for fulltext
        """
        logger.info(f"Running `search_cypher_query` with vector_query={vector_query is not None}, fulltext_query={fulltext_query is not None}")

        if not vector_query and not fulltext_query:
            raise ToolError("At least one of vector_query or fulltext_query must be provided")

        if _is_write_query(cypher_query):
            raise ToolError("Only read queries are allowed in search_cypher_query")

        try:
            # Prepare parameters
            query_params = dict(params)

            # Generate embedding if vector_query provided (using LiteLLM)
            if vector_query:
                logger.debug(f"Generating embedding for vector_query: {vector_query}")
                embedding_response = litellm.embedding(
                    model=embedding_model,
                    input=[vector_query]
                )
                query_params["vector_embedding"] = embedding_response.data[0]["embedding"]

            # Add fulltext query if provided
            if fulltext_query:
                query_params["fulltext_text"] = fulltext_query

            # Execute Cypher query
            query_obj = Query(cypher_query, timeout=float(read_timeout))
            results = await neo4j_driver.execute_query(
                query_obj,
                parameters_=query_params,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Search cypher query returned {len(results)} rows")

            # Layer 3: Sanitize large lists and strings (embeddings, images, text, etc.)
            sanitized_results = [_value_sanitize(el, MAX_LIST_SIZE, MAX_STRING_SIZE) for el in results]

            # Layer 4: Truncate results to stay under token limit
            original_count = len(sanitized_results)
            sanitized_results, was_truncated = _truncate_results_to_token_limit(
                sanitized_results, RESPONSE_TOKEN_LIMIT
            )

            if was_truncated:
                logger.warning(
                    f"Search cypher results truncated from {original_count} to {len(sanitized_results)} rows"
                )

            results_json_str = json.dumps(sanitized_results, default=str)
            return ToolResult(content=[TextContent(type="text", text=results_json_str)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error executing search cypher query: {e}\n{cypher_query}\n{query_params}")
            raise ToolError(f"Neo4j Error: {e}\n{cypher_query}")

        except Exception as e:
            logger.error(f"Error executing search cypher query: {e}\n{cypher_query}")
            raise ToolError(f"Error: {e}")

    @mcp.tool(
        name=namespace_prefix + "read_node_image",
        annotations=ToolAnnotations(
            title="Read Node Image",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_node_image(
        node_element_id: str = Field(..., description="The elementId of the node to read."),
        image_property: Optional[str] = Field(
            None,
            description='Property name holding the base64-encoded image. Defaults to "imageBase64".'
        ),
        mime_type: Optional[str] = Field(
            None,
            description='Override MIME type (e.g., "image/jpeg"). If not provided, reads from node\'s imageMimeType property, falling back to "image/png".'
        ),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of additional text properties to return alongside the image. If not specified, returns all properties (sanitized, excluding the image property).'
        ),
    ) -> list[ToolResult]:
        """
        Retrieve a base64 image stored on a Neo4j node, plus selected text properties.

        Returns a mixed ToolResult with TextContent (node properties) and ImageContent (the image).
        The image property itself is excluded from the text content to avoid duplication.

        **Usage:**
        - node_element_id: get from vector_search/fulltext_search/read_neo4j_cypher (nodeId field)
        - image_property: property storing the base64 image (default: "imageBase64")
        - mime_type: override detected MIME type (default: read from node's imageMimeType, else "image/png")
        - return_properties: comma-separated list of properties to include in text response
        """
        img_prop = image_property or "imageBase64"
        logger.info(f"Running `read_node_image` for node {node_element_id}, image_property={img_prop}")

        try:
            rows = await neo4j_driver.execute_query(
                "MATCH (n) WHERE elementId(n) = $id RETURN labels(n) as labels, properties(n) as props",
                parameters_={"id": node_element_id},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            if not rows:
                raise ToolError(f"Node with elementId '{node_element_id}' not found.")

            row = rows[0]
            labels = row["labels"]
            props: dict[str, Any] = row["props"]

            # Extract image data
            image_b64 = props.get(img_prop)
            if not image_b64:
                raise ToolError(f"Property '{img_prop}' not found or empty on node {node_element_id}.")

            # Determine MIME type
            effective_mime = mime_type or props.get("imageMimeType") or "image/png"

            # Build text content
            text_props: dict[str, Any] = {"nodeId": node_element_id, "labels": labels}
            if return_properties:
                property_list = [p.strip() for p in return_properties.split(",")]
                for p in property_list:
                    if p in props:
                        text_props[p] = props[p]
            else:
                # Return all props sanitized, excluding image property
                filtered = {k: v for k, v in props.items() if k != img_prop}
                text_props.update(_value_sanitize(filtered, MAX_LIST_SIZE, MAX_STRING_SIZE))

            text_json = json.dumps(text_props, default=str)

            return ToolResult(content=[
                TextContent(type="text", text=text_json),
                ImageContent(type="image", data=image_b64, mimeType=effective_mime),
            ])

        except ToolError:
            raise

        except Neo4jError as e:
            logger.error(f"Neo4j Error reading node image: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error reading node image: {e}")
            raise ToolError(f"Error: {e}")

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    embedding_model: str = "text-embedding-3-small",
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    read_timeout: int = 30,
    schema_sample_size: int = 1000,
) -> None:
    """Main entry point for the Neo4j GraphRAG MCP Server."""
    logger.info("Starting Neo4j GraphRAG MCP Server")
    logger.info(f"Using embedding model: {embedding_model}")

    neo4j_driver = AsyncGraphDatabase.driver(db_url, auth=(username, password))

    mcp = create_mcp_server(
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        database=database,
        namespace=namespace,
        read_timeout=read_timeout,
        config_sample_size=schema_sample_size,
    )

    match transport:
        case "http":
            logger.info(f"Running Neo4j GraphRAG MCP Server with HTTP transport on {host}:{port}...")
            await mcp.run_http_async(host=host, port=port, path=path)
        case "stdio":
            logger.info("Running Neo4j GraphRAG MCP Server with stdio transport...")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(f"Running Neo4j GraphRAG MCP Server with SSE transport on {host}:{port}...")
            await mcp.run_http_async(host=host, port=port, path=path, transport="sse")
        case _:
            logger.error(f"Invalid transport: {transport}")
            raise ValueError(f"Invalid transport: {transport} | Must be 'stdio', 'sse', or 'http'")


if __name__ == "__main__":
    # This file should not be run directly. Use the CLI: mcp-neo4j-graphrag
    # Or import and call: from mcp_neo4j_graphrag import main
    raise RuntimeError(
        "This module should not be run directly. "
        "Use the CLI command: mcp-neo4j-graphrag --help"
    )

