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
        with patch(
            "mcp_neo4j_graphrag.pageindex._query", AsyncMock(side_effect=responses)
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
        self.assertEqual("a* OR b", query.call_args.kwargs["query"])
        self.assertNotIn("STARTS WITH", query.call_args.args[3])

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
        self.assertIn("HAS_SUBSECTION*0..32", query.call_args_list[2].args[3])

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
