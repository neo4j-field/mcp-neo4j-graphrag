"""Opt-in empty-database acceptance test for external ingestion tool discovery."""

import asyncio
import os
import unittest
from uuid import uuid4

from fastmcp import Client
from fastmcp.client.messages import MessageHandler
from neo4j import AsyncGraphDatabase

from mcp_neo4j_graphrag.server import create_mcp_server


class Changes(MessageHandler):
    def __init__(self):
        self.events = asyncio.Queue()

    async def on_tool_list_changed(self, message):
        self.events.put_nowait(message)


@unittest.skipUnless(os.getenv("NEO4J_MCP_RUN_INTEGRATION") == "1", "requires disposable empty database")
class ExternalIngestRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_to_sections_to_deleted_notifies_without_reconnect(self):
        driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        database = os.environ["NEO4J_DATABASE"]
        run = uuid4().hex
        names = {"list_retrieval_documents", "get_document_content", "get_document_outline", "get_section_content"}
        try:
            async with driver.session(database=database) as external:
                count = await (await external.run("MATCH (n) RETURN count(n) AS n")).single()
                self.assertEqual(0, count["n"], "Use an empty disposable database")
                handler = Changes()
                server = create_mcp_server(driver, "unused", database=database)
                async with Client(server, message_handler=handler) as client:
                    self.assertFalse(names & {t.name for t in await client.list_tools()})
                    result = await external.run(
                        "CREATE (d:Document {id:$id,active:true})-[:HAS_SECTION]->"
                        "(s:Section {id:$id})-[:HAS_ELEMENT]->(e:Element {id:$id,text:'Maintenance'})",
                        id=run,
                    )
                    await result.consume()
                    # No GraphRAG call between external write and notification.
                    await asyncio.wait_for(handler.events.get(), 10)
                    self.assertTrue(names <= {t.name for t in await client.list_tools()})
                    result = await external.run("MATCH (n {id:$id}) DETACH DELETE n", id=run)
                    await result.consume()
                    await asyncio.wait_for(handler.events.get(), 10)
                    self.assertFalse(names & {t.name for t in await client.list_tools()})
        finally:
            async with driver.session(database=database) as cleanup:
                await (await cleanup.run("MATCH (n {id:$id}) DETACH DELETE n", id=run)).consume()
            await driver.close()
