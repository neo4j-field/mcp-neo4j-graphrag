"""New retrieval tools are reachable through the real GraphRAG server/config."""

import argparse
import os
import unittest
from unittest.mock import MagicMock, patch

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

    async def test_real_server_registers_namespaced_read_only_tools_without_database_work(
        self,
    ):
        driver = MagicMock()
        server = create_mcp_server(driver, "configured-model", namespace="poc")
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
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
