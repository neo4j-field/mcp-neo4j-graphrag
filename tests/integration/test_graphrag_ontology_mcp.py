"""Read-only domain retrieval against UUID fixtures in an isolated test database."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastmcp import Client
from neo4j import AsyncGraphDatabase

from mcp_neo4j_graphrag.server import create_mcp_server
from mcp_neo4j_graphrag.ontology import TEMPLATE_EXTENSION
from tests.unit.test_graphrag_ontology import ontology_fixture

DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


@unittest.skipUnless(
    os.getenv("NEO4J_MCP_RUN_INTEGRATION") == "1", "requires disposable Neo4j"
)
class GraphRAGOntologyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self.marker = uuid4().hex
        self.ontology_name = "ontology-retrieval-" + self.marker
        self.names = {
            "person": "RetrievalPerson_" + self.marker,
            "movie": "RetrievalMovie_" + self.marker,
            "rel": "RETRIEVAL_CREDIT_" + self.marker,
            "aliases": "RetrievalAliases_" + self.marker,
            "blocklist": "RetrievalBlocked_" + self.marker,
        }
        await self.query(
            """CREATE (o:Ontology {name:$ontology, version:'poc1',description:'A private extraction ontology',testMarker:$marker})
            CREATE (p:NodeType {name:$person,description:'A performer',testMarker:$marker})
            CREATE (m:NodeType {name:$movie,description:'A film',testMarker:$marker})
            CREATE (r:RelationshipType {name:$rel,description:'A performance credit',testMarker:$marker})
            CREATE (o)-[:CONTAINS]->(p), (o)-[:CONTAINS]->(m), (o)-[:CONTAINS]->(r), (r)-[:FROM]->(p), (r)-[:TO]->(m)
            CREATE (p)-[:HAS_PROPERTY]->(pd:PropertyDef {name:'name',type:'STRING',is_key:true,description:'Canonical performer name',normalizers:['whitespace','lowercase','alias_map','blocklist'],testMarker:$marker})
            CREATE (m)-[:HAS_PROPERTY]->(:PropertyDef {name:'title',type:'STRING',required:true,testMarker:$marker})
            CREATE (r)-[:HAS_PROPERTY]->(:PropertyDef {name:'role',type:'STRING',testMarker:$marker})
            CREATE (o)-[:DEFINES]->(a:AliasMap {name:$aliases,testMarker:$marker})
            CREATE (a)-[:HAS_ALIAS]->(:Alias {from:'bob',to:'robert',testMarker:$marker})
            CREATE (o)-[:DEFINES]->(b:Blocklist {name:$blocklist,testMarker:$marker})
            CREATE (b)-[:HAS_TERM]->(:BlockedTerm {value:'unknown',testMarker:$marker})
            CREATE (pd)-[:USES_ALIAS_MAP]->(a), (pd)-[:USES_BLOCKLIST]->(b)""",
            ontology=self.ontology_name,
            marker=self.marker,
            **self.names,
        )

    async def query(self, query, **params):
        async with self.driver.session(database=DATABASE) as session:
            return await (await session.run(query, **params)).data()

    async def asyncTearDown(self):
        await self.query(
            "MATCH (n {testMarker:$marker}) DETACH DELETE n", marker=self.marker
        )
        await self.driver.close()

    async def retrieve(self, args=None, **config):
        server = create_mcp_server(
            self.driver, "text-embedding-3-small", database=DATABASE, **config
        )
        async with Client(server) as client:
            result = await client.call_tool("get_ontology", args or {})
            return json.loads(result.content[0].text)

    async def test_existing_ontology_db_default_selection_preserves_semantics_read_only(
        self,
    ):
        count_before = (
            await self.query(
                "MATCH (n {testMarker:$marker}) RETURN count(n) AS n",
                marker=self.marker,
            )
        )[0]["n"]
        result = await self.retrieve(
            ontology_name=self.ontology_name,
            ontology_database=DATABASE,
            ontology_driver=self.driver,
        )
        self.assertEqual("template_ontology_db", result["source"])
        doc = result["ontology"]
        self.assertEqual(self.ontology_name, doc["id"])
        self.assertEqual("poc1", doc["version"])
        person = doc["nodes"][self.names["person"]]
        self.assertEqual("A performer", person["description"])
        cfg = person["properties"]["name"]["extensions"]["custom"][TEMPLATE_EXTENSION]
        self.assertEqual(
            ["whitespace", "lowercase", "alias_map", "blocklist"], cfg["normalizers"]
        )
        self.assertEqual(self.names["aliases"], cfg["alias_map_name"])
        self.assertEqual(
            {"bob": "robert"},
            doc["extensions"]["custom"][TEMPLATE_EXTENSION]["alias_maps"][
                self.names["aliases"]
            ],
        )
        self.assertEqual(
            {"node": self.names["movie"]}, doc["relationships"][self.names["rel"]]["to"]
        )
        self.assertEqual(
            count_before,
            (
                await self.query(
                    "MATCH (n {testMarker:$marker}) RETURN count(n) AS n",
                    marker=self.marker,
                )
            )[0]["n"],
        )

    async def test_ambiguity_and_explicit_selection(self):
        other = "other-" + self.marker
        await self.query(
            "CREATE (:Ontology {name:$name,testMarker:$marker})",
            name=other,
            marker=self.marker,
        )
        with self.assertRaisesRegex(Exception, "Several ontologies"):
            await self.retrieve()
        selected = await self.retrieve(
            {"ontology_name": self.ontology_name, "entities": [self.names["rel"]]}
        )
        self.assertEqual(
            {self.names["person"], self.names["movie"]},
            set(selected["ontology"]["nodes"]),
        )
        self.assertEqual(
            {self.names["rel"]}, set(selected["ontology"]["relationships"])
        )

    async def test_file_source_takes_precedence_over_db(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.yaml"
            path.write_text(json.dumps(ontology_fixture()))
            result = await self.retrieve(
                {"entities": ["FeaturedIn"]}, ontology_path=str(path)
            )
            self.assertEqual("file", result["source"])
            self.assertEqual("film-model", result["ontology_id"])
            self.assertEqual(
                20, result["ontology"]["relationships"]["credits"]["to"]["max_count"]
            )

    async def test_dangling_ontology_db_endpoint_is_actionable_error(self):
        await self.query(
            "MATCH (r:RelationshipType {name:$name})-[edge:TO]->() DELETE edge",
            name=self.names["rel"],
        )
        with self.assertRaisesRegex(Exception, "requires one source and target"):
            await self.retrieve(ontology_name=self.ontology_name)
