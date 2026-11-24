"""Opt-in isolated graph tests for model-free document navigation."""

import json
import logging
import os
import unittest
from uuid import uuid4

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from neo4j import AsyncGraphDatabase
from mcp_neo4j_graphrag.pageindex import register_pageindex_tools

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


@unittest.skipUnless(
    os.getenv("NEO4J_MCP_RUN_INTEGRATION") == "1", "opt-in disposable graph test"
)
class PageIndexRetrievalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.run_id = uuid4().hex
        self.doc = "retrieval-" + self.run_id
        self.section = self.doc + "_sec_0"
        self.child = self.doc + "_sec_1"
        self.flat = self.doc + "-flat"
        async with self.driver.session(database=self.database) as session:
            await session.run(
                """
                CREATE (d:Document {id:$doc,name:'Navigation fixture',active:true,source:'/fixture.pdf',testRunId:$run})
                CREATE (s:Section {id:$section,title:'Overview',level:1,order:0,testRunId:$run})
                CREATE (child:Section {id:$child,title:'Details',level:2,order:1,testRunId:$run})
                CREATE (d)-[:HAS_SECTION]->(s), (d)-[:HAS_SECTION]->(child),(s)-[:HAS_SUBSECTION]->(child)
                CREATE (p:Page {id:$doc+'-page',pageNumber:0,testRunId:$run}),(d)-[:HAS_PAGE]->(p)
                CREATE (e:Element {id:$doc+'-element-ten',order:0,text:'abcdefghij',pageNumber:0,testRunId:$run})
                CREATE (e2:Element {id:$doc+'-element-two',order:1,text:'SECOND',pageNumber:0,testRunId:$run})
                CREATE (s)-[:HAS_ELEMENT]->(e),(child)-[:HAS_ELEMENT]->(e2),
                       (p)-[:HAS_ELEMENT]->(e),(p)-[:HAS_ELEMENT]->(e2)
                CREATE (partial:Chunk {id:$doc+'-partial',active:true,text:'WRONG PARTIAL CHUNK',testRunId:$run}),
                       (partial)-[:PART_OF]->(d),(partial)-[:HAS_ELEMENT]->(e)
                CREATE (flat:Document {id:$flat,name:'Heading-free',active:true,testRunId:$run})
                CREATE (c:Chunk {id:$flat+'-chunk',active:true,text:'Fallback text',index:0,testRunId:$run}),
                       (c)-[:PART_OF]->(flat)
                CREATE (visual:Document {id:$doc+'-visual',name:'Slides',parseMode:'page_image',active:true,testRunId:$run})
                CREATE (slide:Page {id:$doc+'-slide',pageNumber:1,textDescription:'Diagram description',testRunId:$run}),
                       (visual)-[:HAS_PAGE]->(slide)
                CREATE (emptyChunk:Chunk {id:$doc+'-empty',active:true,text:'',testRunId:$run}),
                       (emptyChunk)-[:PART_OF]->(visual)
                CREATE (inactive:Document {id:$doc+'-inactive',active:false,testRunId:$run})
                CREATE (old:Page {id:$doc+'-old-page',active:true,text:'Hidden',testRunId:$run}),
                       (inactive)-[:HAS_PAGE]->(old)
            """,
                doc=self.doc,
                section=self.section,
                child=self.child,
                flat=self.flat,
                run=self.run_id,
            )
        mcp = FastMCP("retrieval-integration")
        register_pageindex_tools(mcp, self.driver, self.database, "", 7)
        self.client = Client(mcp)
        await self.client.__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__(None, None, None)
        async with self.driver.session(database=self.database) as session:
            await session.run(
                "MATCH (n {testRunId:$run}) DETACH DELETE n", run=self.run_id
            )
        await self.driver.close()

    async def call(self, name, **args):
        response = await self.client.call_tool(name, args)
        return json.loads(response.content[0].text)

    async def test_outline_root_detection_and_drilldown(self):
        result = await self.call("get_document_outline", document_id=self.doc, depth=1)
        self.assertEqual([self.section], [s["section_id"] for s in result["items"]])
        self.assertTrue(result["items"][0]["has_children"])
        exhausted = await self.call(
            "get_document_outline", document_id=self.doc, offset=100
        )
        self.assertTrue(exhausted["structure_available"])
        self.assertTrue(exhausted["page_empty"])
        self.assertEqual("get_section_content", exhausted["content_tool"])
        result = await self.call(
            "get_document_outline",
            document_id=self.doc,
            root_section_id=self.child,
            depth=1,
        )
        self.assertEqual(
            [self.section, self.child],
            [s["section_id"] for s in result["items"][0]["path"]],
        )

    async def test_original_complete_text_resumable_with_precise_page(self):
        arguments = {
            "document_id": self.doc,
            "section_id": self.section,
            "max_chars": 4,
        }
        texts = []
        for _ in range(10):
            result = await self.call("get_section_content", **arguments)
            texts.extend(item["text"] for item in result["items"])
            self.assertTrue(
                all(item["citation"]["page_number"] == 1 for item in result["items"])
            )
            self.assertFalse(result["fallback_used"])
            if result["next"] is None:
                break
            arguments.update(result["next"])
        else:
            self.fail("content pagination did not terminate")
        self.assertEqual("abcdefghijSECOND", "".join(texts))
        direct = await self.call(
            "get_section_content",
            document_id=self.doc,
            section_id=self.section,
            include_descendants=False,
        )
        self.assertEqual("abcdefghij", "".join(i["text"] for i in direct["items"]))

    async def test_heading_free_fallback_and_active_gate(self):
        outline = await self.call("get_document_outline", document_id=self.flat)
        self.assertFalse(outline["structure_available"])
        result = await self.call("get_document_content", document_id=self.flat)
        self.assertTrue(result["fallback_used"])
        self.assertEqual("Fallback text", result["items"][0]["text"])
        self.assertIsNone(result["items"][0]["citation"]["page_number"])
        with self.assertRaises(ToolError):
            await self.call("get_document_content", document_id=self.doc + "-inactive")
        with self.assertRaises(ToolError):
            await self.call(
                "get_section_content", document_id=self.flat, section_id=self.section
            )

    async def test_page_image_reads_page_description_instead_of_empty_chunk(self):
        result = await self.call(
            "get_document_content", document_id=self.doc + "-visual"
        )
        self.assertEqual("Diagram description", result["items"][0]["text"])
        self.assertEqual(2, result["items"][0]["citation"]["page_number"])
        self.assertTrue(result["items"][0]["node_element_id"])

    async def test_discovery_literal_query_and_pagination(self):
        result = await self.call("list_retrieval_documents", query=self.run_id, limit=1)
        self.assertEqual(1, len(result["items"]))
        self.assertTrue(result["has_more"])
        next_page = await self.call(
            "list_retrieval_documents",
            query=self.run_id,
            limit=1,
            offset=result["next_offset"],
        )
        self.assertNotEqual(
            result["items"][0]["document_id"], next_page["items"][0]["document_id"]
        )
