from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fastmcp.exceptions import ToolError

from mcp_neo4j_graphrag.server import _parse_json_object, create_mcp_server


def _assert_no_open_object_schemas(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is not True
        for value in schema.values():
            _assert_no_open_object_schemas(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_no_open_object_schemas(value)


@pytest.mark.asyncio
async def test_cypher_parameters_are_json_strings_for_strict_tool_consumers():
    server = create_mcp_server(AsyncMock(), embedding_model="test-model")

    tools = await server.get_tools()

    pre_filter_schema = tools["vector_search"].parameters["properties"]["pre_filter"]
    assert {variant.get("type") for variant in pre_filter_schema["anyOf"]} == {
        "string",
        "null",
    }

    for name in ("read_neo4j_cypher", "write_neo4j_cypher", "search_cypher_query"):
        params_schema = tools[name].parameters["properties"]["params"]
        assert params_schema["default"] is None
        assert {variant.get("type") for variant in params_schema["anyOf"]} == {
            "string",
            "null",
        }

    for tool in tools.values():
        _assert_no_open_object_schemas(tool.parameters)


def test_cypher_parameter_json_must_decode_to_an_object():
    assert _parse_json_object(None, "Cypher params") == {}
    assert _parse_json_object('{"year": 2020}', "Cypher params") == {"year": 2020}

    with pytest.raises(ToolError, match="valid JSON object"):
        _parse_json_object("not-json", "Cypher params")
    with pytest.raises(ToolError, match="decode to a JSON object"):
        _parse_json_object("[]", "Cypher params")


@pytest.mark.asyncio
async def test_read_neo4j_cypher_decodes_non_null_params_for_driver():
    driver = AsyncMock()
    driver.execute_query.return_value = []
    server = create_mcp_server(driver, embedding_model="test-model")
    tool = (await server.get_tools())["read_neo4j_cypher"]

    await tool.run(
        {
            "query": "MATCH (n) WHERE n.year = $year RETURN n",
            "params": '{"year": 2020}',
        }
    )

    assert driver.execute_query.await_args.kwargs["parameters_"] == {"year": 2020}


@pytest.mark.asyncio
async def test_search_cypher_query_decodes_non_null_params_for_driver(monkeypatch):
    driver = AsyncMock()
    driver.execute_query.return_value = []
    monkeypatch.setattr(
        "mcp_neo4j_graphrag.server.litellm.embedding",
        lambda **_kwargs: SimpleNamespace(data=[{"embedding": [0.1, 0.2]}]),
    )
    server = create_mcp_server(driver, embedding_model="test-model")
    tool = (await server.get_tools())["search_cypher_query"]

    await tool.run(
        {
            "cypher_query": "RETURN $year, $vector_embedding",
            "vector_query": "mesembrine",
            "params": '{"year": 2020}',
        }
    )

    assert driver.execute_query.await_args.kwargs["parameters_"] == {
        "year": 2020,
        "vector_embedding": [0.1, 0.2],
    }


@pytest.mark.asyncio
async def test_vector_search_decodes_pre_filter_into_prefixed_driver_parameters(monkeypatch):
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        [{"properties": ["embedding"]}],
        [],
    ]
    monkeypatch.setattr(
        "mcp_neo4j_graphrag.server.litellm.embedding",
        lambda **_kwargs: SimpleNamespace(data=[{"embedding": [0.1, 0.2]}]),
    )
    server = create_mcp_server(driver, embedding_model="test-model")
    tool = (await server.get_tools())["vector_search"]

    await tool.run(
        {
            "text_query": "mesembrine",
            "vector_index": "chunk_embedding_vector",
            "pre_filter": '{"documentName": "source.pdf", "year": 2020}',
        }
    )

    search_call = driver.execute_query.await_args_list[1]
    assert search_call.kwargs["parameters_"] == {
        "index_name": "chunk_embedding_vector",
        "fetch_k": 100,
        "top_k": 5,
        "query_vector": [0.1, 0.2],
        "pf_documentName": "source.pdf",
        "pf_year": 2020,
    }
    search_query = search_call.args[0]
    assert "node.documentName = $pf_documentName" in search_query
    assert "node.year = $pf_year" in search_query
