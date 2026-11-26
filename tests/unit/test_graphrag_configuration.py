"""New retrieval tools are reachable through the real GraphRAG server/config."""

import argparse
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import Client
from mcp_neo4j_graphrag.server import create_mcp_server
from mcp_neo4j_graphrag.utils import process_config


class GraphRAGConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **overrides):
        values = dict.fromkeys(
            [
                "db_url",
                "username",
                "password",
                "database",
                "embedding_model",
                "namespace",
                "transport",
                "server_host",
                "server_port",
                "server_path",
                "read_timeout",
                "schema_sample_size",
                "ontology_file",
                "ontology_name",
            ]
        )
        return argparse.Namespace(**(values | overrides))

    def test_configured_ontology_file_and_separate_database(self):
        with patch.dict(
            os.environ,
            {
                "ONTOLOGY_FILE": "/configured/domain.yaml",
                "ONTOLOGY_NAME": "domain",
                "NEO4J_ONTOLOGY_URI": "bolt://ontology:7687",
                "NEO4J_ONTOLOGY_DATABASE": "ontologies",
            },
            clear=True,
        ):
            config = process_config(self.args())
            self.assertEqual("/configured/domain.yaml", config["ontology_file"])
            self.assertEqual("domain", config["ontology_name"])
            self.assertEqual("bolt://ontology:7687", config["ontology_db_url"])
            self.assertEqual("ontologies", config["ontology_database"])
            override = process_config(
                self.args(ontology_file="/explicit.json", ontology_name="explicit")
            )
            self.assertEqual("/explicit.json", override["ontology_file"])
            self.assertEqual("explicit", override["ontology_name"])

    async def test_real_server_registers_namespaced_read_only_tools_after_topology_inspection(
        self,
    ):
        driver = MagicMock()
        server = create_mcp_server(driver, "configured-model", namespace="poc")
        with patch(
            "mcp_neo4j_graphrag.pageindex._query",
            AsyncMock(return_value=[{"documents": True, "sections": True}]),
        ) as inspect_topology:
            async with Client(server) as client:
                tools = {tool.name: tool for tool in await client.list_tools()}
        inspect_topology.assert_awaited_once()
        for name in (
            "get_ontology",
            "list_retrieval_documents",
            "get_document_outline",
            "get_section_content",
            "get_document_content",
        ):
            self.assertIn("poc-" + name, tools)
            self.assertTrue(tools["poc-" + name].annotations.readOnlyHint)
        driver.session.assert_not_called()
        driver.execute_query.assert_not_called()

    async def test_real_server_keeps_ontology_and_cypher_when_navigation_is_hidden(
        self,
    ):
        server = create_mcp_server(MagicMock(), "configured-model", namespace="poc")
        with patch(
            "mcp_neo4j_graphrag.pageindex._query",
            AsyncMock(return_value=[{"documents": False, "sections": False}]),
        ):
            async with Client(server) as client:
                names = {tool.name for tool in await client.list_tools()}
        self.assertIn("poc-get_ontology", names)
        self.assertIn("poc-read_neo4j_cypher", names)
        self.assertFalse(
            names
            & {
                "poc-list_retrieval_documents",
                "poc-get_document_content",
                "poc-get_document_outline",
                "poc-get_section_content",
            }
        )
