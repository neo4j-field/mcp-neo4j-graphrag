"""Public MCP notification contract; ingestion changes independently of requests."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Client, FastMCP
from fastmcp.client.messages import MessageHandler

from mcp_neo4j_graphrag.pageindex import _PageIndexDiscovery, register_pageindex_tools


class Changes(MessageHandler):
    def __init__(self):
        self.events = asyncio.Queue()

    async def on_tool_list_changed(self, message):
        self.events.put_nowait(message)


class RefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_changes_notify_two_clients_without_tool_calls(self):
        state = {"documents": False, "sections": False}
        discovery = _PageIndexDiscovery(MagicMock(), "neo4j", "", 7, poll_interval=.02)
        server = FastMCP("refresh-test", lifespan=discovery.lifespan)
        register_pageindex_tools(server, discovery.driver, "neo4j", discovery=discovery)
        first, second = Changes(), Changes()
        with patch("mcp_neo4j_graphrag.pageindex._query", AsyncMock(side_effect=lambda *a, **kw: [dict(state)])):
            async with Client(server, message_handler=first) as a:
                self.assertEqual([], await a.list_tools())
                async with Client(server, message_handler=second) as b:
                    self.assertEqual([], await b.list_tools())
                    state.update(documents=True, sections=True)
                    await asyncio.wait_for(first.events.get(), 2)
                    await asyncio.wait_for(second.events.get(), 2)
                    self.assertEqual(4, len(await a.list_tools()))
                    self.assertEqual(4, len(await b.list_tools()))
                    await asyncio.sleep(.08)
                    self.assertTrue(first.events.empty())
                    self.assertTrue(second.events.empty())
                    state.update(documents=False, sections=False)
                    await asyncio.wait_for(first.events.get(), 2)
                    await asyncio.wait_for(second.events.get(), 2)
                    self.assertEqual([], await a.list_tools())
                    self.assertEqual([], await b.list_tools())
            self.assertFalse(discovery._sessions)

    async def test_disconnected_session_does_not_break_other_notifications(self):
        import anyio
        discovery = _PageIndexDiscovery(MagicMock(), "neo4j", "", 7)
        closed, active = MagicMock(), MagicMock()
        closed.send_tool_list_changed = AsyncMock(side_effect=anyio.ClosedResourceError)
        active.send_tool_list_changed = AsyncMock()
        discovery._sessions[closed] = set()
        discovery._sessions[active] = set()
        discovery.enabled = AsyncMock(return_value={"get_document_content"})
        await discovery._refresh_sessions()
        active.send_tool_list_changed.assert_awaited_once()
        self.assertNotIn(closed, discovery._sessions)
        await discovery._refresh_sessions()
        active.send_tool_list_changed.assert_awaited_once()

    async def test_unexpected_notification_error_does_not_affect_other_clients(self):
        discovery = _PageIndexDiscovery(MagicMock(), "neo4j", "", 7)
        broken, active = MagicMock(), MagicMock()
        broken.send_tool_list_changed = AsyncMock(side_effect=RuntimeError("closed transport"))
        active.send_tool_list_changed = AsyncMock()
        discovery._sessions[broken] = set()
        discovery._sessions[active] = set()
        discovery.enabled = AsyncMock(return_value={"get_document_content"})
        with self.assertLogs("mcp_neo4j_graphrag.pageindex", level="ERROR"):
            await discovery._refresh_sessions()
        active.send_tool_list_changed.assert_awaited_once()
        self.assertNotIn(broken, discovery._sessions)

    async def test_transient_probe_failure_retains_advertised_capabilities(self):
        from neo4j.exceptions import ServiceUnavailable
        discovery = _PageIndexDiscovery(MagicMock(), "neo4j", "", 7)
        session = MagicMock()
        session.send_tool_list_changed = AsyncMock()
        expected = {"get_document_content"}
        discovery._sessions[session] = expected
        with patch("mcp_neo4j_graphrag.pageindex._query", AsyncMock(side_effect=ServiceUnavailable("offline"))):
            with self.assertRaises(ServiceUnavailable):
                await discovery._refresh_sessions()
        self.assertEqual(expected, discovery._sessions[session])
        session.send_tool_list_changed.assert_not_awaited()

    async def test_observer_recovers_from_unexpected_failure(self):
        import anyio
        discovery = _PageIndexDiscovery(MagicMock(), "neo4j", "", 7, poll_interval=.01)
        session = MagicMock()
        discovery._sessions[session] = set()
        recovered = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("unexpected probe failure")
            recovered.set()

        discovery._refresh_sessions = refresh
        with self.assertLogs("mcp_neo4j_graphrag.pageindex", level="ERROR"):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(discovery._observe)
                await asyncio.wait_for(recovered.wait(), 2)
                tasks.cancel_scope.cancel()
        self.assertGreaterEqual(calls, 2)
