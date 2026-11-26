"""Declared ontology semantics are preserved without inventing physical graph state."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastmcp import Client
from mcp_neo4j_graphrag.ontology import (
    SCHEMA_URI,
    load_ontology_file,
    ontology_response,
    select_ontology,
    validate_ontology,
    register_ontology_tools,
)
from fastmcp import FastMCP


def ontology_fixture():
    return {
        "schema": SCHEMA_URI,
        "version": "poc-1",
        "id": "film-model",
        "description": "Declared film-industry meaning",
        "nodes": {
            "people": {
                "labels": {"identifier": "Person", "optional": ["Actor"]},
                "aliases": ["Performer"],
                "description": "A film industry professional",
                "properties": {
                    "full-name": {
                        "name": "name",
                        "type": "STRING",
                        "key": True,
                        "description": "Full legal name",
                        "pattern": "^.+$",
                        "aliases": ["legalName"],
                    },
                    "born": {"type": "DATE"},
                    "tags": {"type": "LIST<STRING>", "one_of": ["lead", "supporting"]},
                },
                "constraints": {
                    "person-key": {"type": "KEY", "properties": ["full-name"]}
                },
                "extensions": {"custom": {"external-system": {"meaning": "kept"}}},
            },
            "films": {
                "label": "Movie",
                "aliases": ["Film"],
                "properties": {
                    "title": {"type": "STRING", "must_exist": True},
                },
            },
            "genres": {
                "label": "Genre",
                "description": "Unrelated to selected relationship",
            },
        },
        "relationships": {
            "credits": {
                "type": "ACTED_IN",
                "aliases": ["FeaturedIn"],
                "description": "A performance credit",
                "from": {"node": "people", "min_count": 0},
                "to": {"node": "films", "max_count": 20},
                "properties": {
                    "role": {"type": "STRING", "one_of": ["hero", "villain"]}
                },
            }
        },
        "extensions": {
            "custom": {"external": {"note": "Full normative metadata preserved"}}
        },
    }


class GraphRAGOntologyTests(unittest.IsolatedAsyncioTestCase):
    def test_full_normative_semantics_preserved_without_extraction_limitations(self):
        doc = ontology_fixture()
        validate_ontology(doc)
        result = ontology_response(doc, "file")
        self.assertEqual(doc, result["ontology"])
        self.assertIn("not proof of realized database state", result["context"])
        self.assertEqual(
            "DATE", result["ontology"]["nodes"]["people"]["properties"]["born"]["type"]
        )
        self.assertEqual(
            ["hero", "villain"],
            result["ontology"]["relationships"]["credits"]["properties"]["role"][
                "one_of"
            ],
        )

    def test_schema_card_distinguishes_optional_and_implied_labels(self):
        doc = ontology_fixture()
        doc["nodes"]["people"]["labels"]["optional"] = ["Employee"]
        doc["nodes"]["people"]["labels"]["implied"] = ["Party"]
        result = ontology_response(doc, "file", entities=["Employee"])
        self.assertIn('identifying_label="Person"', result["schema_card"])
        self.assertIn('optional_labels=["Employee"]', result["schema_card"])
        self.assertIn('implied_labels=["Party"]', result["schema_card"])
        self.assertEqual({"people"}, set(result["ontology"]["nodes"]))

    def test_relationship_alias_selection_closes_endpoints_and_preserves_cardinality(
        self,
    ):
        doc = ontology_fixture()
        selected = select_ontology(doc, entities=["FEATUREDIN"])
        self.assertEqual({"people", "films"}, set(selected["nodes"]))
        self.assertEqual({"credits"}, set(selected["relationships"]))
        self.assertEqual(20, selected["relationships"]["credits"]["to"]["max_count"])
        self.assertEqual(
            {"people"}, set(select_ontology(doc, entities=["Performer"])["nodes"])
        )
        self.assertEqual(
            {"people"}, set(select_ontology(doc, search="professional")["nodes"])
        )
        self.assertFalse(
            select_ontology(doc, entities=["Performer"], search="unrelated")["nodes"]
        )
        self.assertEqual(doc, ontology_fixture())

    def test_connecting_relationships_and_label_only_endpoints(self):
        doc = ontology_fixture()
        doc["relationships"]["credits"]["from"] = {"label": "Person", "count": 1}
        selected = select_ontology(doc, entities=["Performer", "Film"])
        self.assertEqual({"credits"}, set(selected["relationships"]))
        self.assertEqual({"people", "films"}, set(selected["nodes"]))

    def test_property_overview_does_not_mutate_original_document(self):
        doc = ontology_fixture()
        original = copy.deepcopy(doc)
        result = ontology_response(doc, "file", include_properties=False)
        self.assertFalse(result["properties_included"])
        self.assertNotIn("properties", result["ontology"]["nodes"]["people"])
        self.assertEqual(original, doc)

    def test_custom_metadata_requires_json_compatible_finite_values(self):
        doc = ontology_fixture()
        doc["extensions"]["custom"]["external"]["invalid"] = float("nan")
        with self.assertRaisesRegex(ValueError, "JSON-compatible finite"):
            validate_ontology(doc)

    def test_unknown_endpoint_and_wrong_schema_rejected(self):
        doc = ontology_fixture()
        doc["relationships"]["credits"]["from"]["node"] = "missing"
        with self.assertRaisesRegex(ValueError, "relationships.credits.from"):
            validate_ontology(doc)
        doc = ontology_fixture()
        doc["schema"] = "https://example.org/untrusted/schema.json"
        with self.assertRaisesRegex(ValueError, "pinned schema"):
            validate_ontology(doc)

    def test_file_json_yaml_duplicate_and_alias_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.json"
            path.write_text(json.dumps(ontology_fixture()))
            self.assertEqual(ontology_fixture(), load_ontology_file(path))
            path.write_text("id: a\nid: b")
            with self.assertRaisesRegex(ValueError, "unique strings"):
                load_ontology_file(path)
            path.write_text("recursive: &loop [*loop]")
            with self.assertRaisesRegex(ValueError, "aliases are unsupported"):
                load_ontology_file(path)

    def test_budget_fallback_and_no_match_are_bounded_and_actionable(self):
        doc = ontology_fixture()
        doc["nodes"]["people"]["description"] = "long semantics " * 1000
        with patch("mcp_neo4j_graphrag.ontology.RESPONSE_CHAR_BUDGET", 1800):
            result = ontology_response(doc, "file")
            self.assertEqual("index", result["mode"])
            self.assertNotIn("ontology", result)
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 1800)
            self.assertIn("entities", result["hint"])
            no_match = ontology_response(doc, "file", entities=["nonexistent"])
            self.assertFalse(no_match["matched"])
            self.assertTrue(no_match["entities"])
            self.assertIn("No ontology entity", no_match["hint"])
            self.assertLessEqual(len(json.dumps(no_match, ensure_ascii=False)), 1800)

    async def test_mcp_file_precedence_namespace_no_db_or_model_calls(self):
        driver = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.yaml"
            path.write_text(json.dumps(ontology_fixture()))
            mcp = FastMCP("ontology-test")
            register_ontology_tools(
                mcp, driver, "domain", "poc-", 15, ontology_path=str(path)
            )
            async with Client(mcp) as client:
                tools = {tool.name: tool for tool in await client.list_tools()}
                self.assertTrue(tools["poc-get_ontology"].annotations.readOnlyHint)
                result = await client.call_tool(
                    "poc-get_ontology",
                    {"entities": ["Film"], "include_properties": False},
                )
                data = json.loads(result.content[0].text)
                self.assertEqual("file", data["source"])
                self.assertNotIn("properties", data["ontology"]["nodes"]["films"])
            driver.session.assert_not_called()

    async def test_missing_file_fails_with_physical_discovery_fallback(self):
        mcp = FastMCP("ontology-test")
        driver = MagicMock()
        register_ontology_tools(
            mcp, driver, "domain", "", 15, ontology_path="/nonexistent/ontology.yaml"
        )
        async with Client(mcp) as client:
            with self.assertRaisesRegex(Exception, "get_neo4j_schema_and_indexes"):
                await client.call_tool("get_ontology", {})
        driver.session.assert_not_called()
