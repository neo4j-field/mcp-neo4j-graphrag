"""Bounded PageIndex-style MCP navigation without providers or writes."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Client, FastMCP
from neo4j import READ_ACCESS
from mcp_neo4j_graphrag.pageindex import (
    register_pageindex_tools,
    _query,
    _content_page,
    _bounds,
    CAPABILITIES,
)
from fastmcp.exceptions import ToolError

DOC = {"document_id": "doc", "document_name": "Manual", "source": "/manual.pdf"}


def row(text="abcdefgh", length=8, start=0):
    return {
        "element_id": "e",
        "labels": ["Element"],
        "type": "paragraph",
        "page_number": 0,
        "text": text,
        "text_length": length,
        "text_offset": start,
        "section_ids": ["s"],
    }


class ContentPageTests(unittest.TestCase):
    def test_long_element_continuation_and_page_citation(self):
        result = _content_page([row()], DOC, 2, 0, 20, 3)
        self.assertEqual("abc", result["items"][0]["text"])
        self.assertEqual({"offset": 2, "text_offset": 3}, result["next"])
        self.assertEqual(1, result["items"][0]["citation"]["page_number"])
        self.assertEqual(["s"], result["items"][0]["citation"]["section_ids"])

    def test_text_resume_and_element_limit(self):
        result = _content_page([row("defgh", 8, 3), row()], DOC, 2, 3, 1, 8)
        self.assertEqual("defgh", result["items"][0]["text"])
        self.assertEqual({"offset": 3, "text_offset": 0}, result["next"])
        self.assertFalse(_content_page([row()], DOC, 0, 0, 20, 12)["has_more"])

    def test_pageless_source_anchor_and_format_survive_text_continuation(self):
        for source_format in ("markdown", "text", "html", "docx"):
            with self.subTest(source_format=source_format):
                document = {**DOC, "source_format": source_format}
                element = {
                    **row(),
                    "source_anchor": "lines:4-6",
                    "node_element_id": "4:fixture:7",
                }
                result = _content_page([element], document, 2, 0, 20, 3)
                item = result["items"][0]
                self.assertEqual("lines:4-6", item["source_anchor"])
                self.assertEqual("lines:4-6", item["citation"]["source_anchor"])
                self.assertEqual(source_format, item["citation"]["source_format"])
                self.assertIsNone(item["citation"]["page_index"])
                self.assertIsNone(item["citation"]["page_number"])
                self.assertEqual("4:fixture:7", item["node_element_id"])
                self.assertEqual({"offset": 2, "text_offset": 3}, result["next"])

    def test_invalid_bounds_and_offsets(self):
        for values in [(0, 26, 25), (-1, 1, 25), (0, 1, 25, 0, 20001)]:
            with self.assertRaises(ToolError):
                _bounds(*values)
        with self.assertRaises(ToolError):
            _content_page([row("", 2, 3)], DOC, 0, 3, 1, 100)


class RetrievalMCPTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, name, arguments, responses):
        mcp = FastMCP("pageindex-tests")
        register_pageindex_tools(mcp, MagicMock(), "neo4j", "test-", 7)
        response_iterator = iter(responses)

        async def respond(*args, **kwargs):
            if args[3] == CAPABILITIES:
                return [{"documents": True, "sections": True}]
            return next(response_iterator)

        with patch(
            "mcp_neo4j_graphrag.pageindex._query", AsyncMock(side_effect=respond)
        ) as query:
            async with Client(mcp) as client:
                result = await client.call_tool("test-" + name, arguments)
            return json.loads(result.content[0].text), query

    async def test_discovery_paginates_without_lucene_or_prefix_semantics(self):
        result, query = await self.call(
            "list_retrieval_documents",
            {"query": "a* OR b", "limit": 1},
            [[{"document_id": "a"}, {"document_id": "b"}]],
        )
        self.assertEqual(1, result["next_offset"])
        discovery_call = next(
            call for call in query.call_args_list if "query" in call.kwargs
        )
        self.assertEqual("a* OR b", discovery_call.kwargs["query"])
        self.assertNotIn("STARTS WITH", discovery_call.args[3])

    async def test_outline_reports_paths_and_honest_heading_free_fallback(self):
        result, _ = await self.call(
            "get_document_outline", {"document_id": "doc"}, [[DOC], []]
        )
        self.assertFalse(result["structure_available"])
        self.assertEqual("get_document_content", result["content_tool"])

    async def test_section_content_requires_membership_and_returns_paths(self):
        result, query = await self.call(
            "get_section_content",
            {"document_id": "doc", "section_id": "s"},
            [
                [DOC],
                [{"id": "s"}],
                [row()],
                [{"path": [{"section_id": "s", "title": "Details"}]}],
            ],
        )
        self.assertEqual("s", result["section_path"][0]["section_id"])
        self.assertFalse(result["fallback_used"])
        content_calls = [
            call for call in query.call_args_list if call.args[3] != CAPABILITIES
        ]
        self.assertIn("HAS_SUBSECTION*0..32", content_calls[2].args[3])

    async def test_read_sessions_apply_timeout(self):
        session = AsyncMock()
        session.run.return_value.data.return_value = [{"ok": True}]
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = context
        self.assertEqual(
            [{"ok": True}],
            await _query(driver, "isolated", 7, "RETURN $query", query="literal"),
        )
        driver.session.assert_called_once_with(
            database="isolated", default_access_mode=READ_ACCESS
        )
        self.assertEqual(7, session.run.call_args.args[0].timeout)
        self.assertEqual(
            {"query": "literal"}, session.run.call_args.kwargs["parameters"]
        )

    async def test_only_capability_probe_suppresses_schema_notifications(self):
        session = AsyncMock()
        session.run.return_value.data.return_value = [
            {"documents": False, "sections": False}
        ]
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = context
        await _query(driver, "isolated", 7, CAPABILITIES)
        driver.session.assert_called_once_with(
            database="isolated",
            default_access_mode=READ_ACCESS,
            notifications_min_severity="OFF",
        )
        driver.session.reset_mock()
        await _query(driver, "isolated", 7, "MATCH (d:Document) RETURN d.id")
        driver.session.assert_called_once_with(
            database="isolated", default_access_mode=READ_ACCESS
        )


class DynamicDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def server(self):
        server = FastMCP("dynamic-discovery")
        register_pageindex_tools(server, MagicMock(), "neo4j", "poc-", 7)

        @server.tool(name="other_tool")
        async def other_tool() -> str:
            return "available"

        return server

    async def test_same_client_sees_empty_business_flat_hierarchy_and_removal(self):
        state = {"documents": False, "sections": False}

        async def topology(*args, **kwargs):
            self.assertEqual(CAPABILITIES, args[3])
            return [dict(state)]

        with patch(
            "mcp_neo4j_graphrag.pageindex._query", AsyncMock(side_effect=topology)
        ) as query:
            async with Client(self.server()) as client:

                async def names():
                    return {tool.name for tool in await client.list_tools()}

                self.assertEqual({"other_tool"}, await names())
                # A business-only graph supplies no supported lexical topology.
                self.assertEqual({"other_tool"}, await names())
                state.update(documents=True)
                core = {
                    "other_tool",
                    "poc-list_retrieval_documents",
                    "poc-get_document_content",
                }
                self.assertEqual(core, await names())
                # Flat sections suffice; HAS_SUBSECTION is not required.
                state.update(sections=True)
                structured = core | {
                    "poc-get_document_outline",
                    "poc-get_section_content",
                }
                self.assertEqual(structured, await names())
                self.assertEqual(structured, await names())  # nested hierarchy
                state.update(documents=False, sections=False)
                self.assertEqual({"other_tool"}, await names())
                with self.assertRaisesRegex(ToolError, "unavailable"):
                    await client.call_tool(
                        "poc-get_document_content", {"document_id": "deleted"}
                    )
                self.assertEqual(
                    "available",
                    (await client.call_tool("other_tool", {})).content[0].text,
                )
        self.assertGreaterEqual(query.await_count, 7)

    async def test_probe_failure_hides_only_navigation_and_blocks_direct_calls(self):
        from neo4j.exceptions import ServiceUnavailable

        with patch(
            "mcp_neo4j_graphrag.pageindex._query",
            AsyncMock(side_effect=ServiceUnavailable("offline")),
        ):
            with self.assertLogs("mcp_neo4j_graphrag.pageindex", level="WARNING"):
                async with Client(self.server()) as client:
                    self.assertEqual(
                        ["other_tool"], [t.name for t in await client.list_tools()]
                    )
                    with self.assertRaisesRegex(
                        ToolError, "connectivity/read permissions"
                    ):
                        await client.call_tool(
                            "poc-get_document_outline", {"document_id": "d"}
                        )
                    self.assertEqual(
                        "available",
                        (await client.call_tool("other_tool", {})).content[0].text,
                    )
