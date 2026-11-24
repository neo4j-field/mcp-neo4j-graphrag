"""Read domain semantics from a pinned spec artifact or the template Ontology DB.

Inspired by https://github.com/neo4j-field/text2cypher-ontology-agent : ontology
context precedes physical schema sampling. No DBMS ontology procedure, LLM call,
entity-server dependency or runtime schema download is required.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from fastmcp.exceptions import ToolError
from jsonschema import Draft202012Validator
from mcp.types import ToolAnnotations
from neo4j import unit_of_work
from pydantic import Field

SCHEMA_URI = "https://neo4j.com/docs/ontology-graph-spec/1.1.0/schema.json"
TEMPLATE_EXTENSION = "neo4j-mcp-workspace-template"
# A deterministic rough character budget bounds every response, including indexes.
# It is not a promise about any particular model's tokenizer.
RESPONSE_CHAR_BUDGET = 12000
MAX_FILE_BYTES = 5_000_000
_SCHEMA = json.loads(
    (Path(__file__).parent / "schemas" / "ontology-graph-spec-1.1.0.json").read_text()
)
_VALIDATOR = Draft202012Validator(_SCHEMA)
_CONTEXT = (
    "Declared domain semantics, not proof of realized database state. Use this before "
    "writing domain Cypher; use get_neo4j_schema_and_indexes for physical indexes, "
    "populated properties or schema gaps, and read queries for actual values/counts. "
    "Aliases describe meaning and are not automatically database labels."
)


class _DocumentLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError(
                "YAML aliases are unsupported; use explicit JSON-compatible values"
            )
        return super().compose_node(parent, index)


def _mapping(loader, node):
    result = {}
    for key, value in loader.construct_pairs(node, deep=True):
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Ontology mapping keys must be unique strings: {key!r}")
        result[key] = value
    return result


_DocumentLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def _node_labels(node):
    labels = node.get("labels", {})
    return [
        value
        for value in [
            node.get("label"),
            labels.get("identifier"),
            *labels.get("implied", []),
            *labels.get("optional", []),
        ]
        if value
    ]


def _endpoint_ids(endpoint, nodes):
    if endpoint.get("node") is not None:
        nid = endpoint["node"]
        if nid not in nodes:
            raise ValueError(f"Unknown relationship endpoint node id: {nid!r}")
        return {nid}
    if endpoint.get("label"):
        matches = {
            nid
            for nid, node in nodes.items()
            if endpoint["label"] in _node_labels(node)
        }
        if not matches:
            raise ValueError(
                f"Unknown relationship endpoint label: {endpoint['label']!r}"
            )
        return matches
    # The normative spec permits unconstrained endpoints. Do not invent a node.
    return set()


def validate_ontology(document):
    """Accept the full pinned specification, rather than the extraction subset."""
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ontology must contain JSON-compatible finite values") from exc
    errors = sorted(
        _VALIDATOR.iter_errors(document), key=lambda e: str(list(e.absolute_path))
    )
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"{path}: {error.message}")
    if document["schema"] != SCHEMA_URI:
        raise ValueError(f"schema: expected pinned schema URI {SCHEMA_URI}")
    nodes = document.get("nodes", {})
    for rid, rel in document.get("relationships", {}).items():
        for end in ("from", "to"):
            try:
                _endpoint_ids(rel[end], nodes)
            except ValueError as exc:
                raise ValueError(f"relationships.{rid}.{end}: {exc}") from exc
    return document


def load_ontology_file(path):
    """Read only the deployment-configured path; model input cannot choose files."""
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(
            f"Configured ontology exceeds {MAX_FILE_BYTES} bytes; split the artifact"
        )
    try:
        document = yaml.load(content.decode("utf-8"), Loader=_DocumentLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ValueError(f"Invalid UTF-8 YAML/JSON ontology: {exc}") from exc
    return validate_ontology(document)


def _property_document(prop):
    value = {
        key: prop[key] for key in ["type", "description"] if prop.get(key) is not None
    }
    value.setdefault("type", "STRING")
    if prop.get("required"):
        value["must_exist"] = True
    if prop.get("is_key"):
        value["key"] = True
    extra = {
        key: val
        for key, val in prop.items()
        if key not in {"name", "type", "description", "required", "is_key"}
        and val is not None
    }
    if extra:
        value["extensions"] = {"custom": {TEMPLATE_EXTENSION: extra}}
    return value


async def load_ontology_db(driver, database, ontology_name, read_timeout):
    """Read the existing template ontology graph in one read transaction.

    This DB stores an extraction-oriented subset. The returned source marker makes
    that boundary explicit; file mode retains the full normative spec semantics.
    """

    @unit_of_work(timeout=float(read_timeout))
    async def read(tx):
        async def rows(query, **params):
            result = await tx.run(query, **params)
            return await result.data()

        found = await rows(
            "MATCH (o:Ontology) WHERE $name IS NULL OR o.name=$name RETURN properties(o) AS metadata ORDER BY o.name",
            name=ontology_name,
        )
        if not found:
            raise ValueError(
                "No matching Ontology DB ontology. Configure ONTOLOGY_FILE or import an ontology first"
            )
        if len(found) != 1:
            names = [row["metadata"]["name"] for row in found[:20]]
            raise ValueError(
                f"Several ontologies are available; specify ontology_name or configure ONTOLOGY_NAME: {names}"
            )
        metadata = found[0]["metadata"]
        name = metadata["name"]
        nodes = await rows(
            """MATCH (:Ontology {name:$name})-[:CONTAINS]->(n:NodeType)
            OPTIONAL MATCH (n)-[:HAS_PROPERTY]->(p:PropertyDef)
            OPTIONAL MATCH (p)-[:USES_ALIAS_MAP]->(a:AliasMap)
            OPTIONAL MATCH (p)-[:USES_BLOCKLIST]->(b:Blocklist)
            RETURN properties(n) AS metadata,
                   collect(p{.*, alias_map_name:a.name, blocklist_name:b.name}) AS properties ORDER BY metadata.name""",
            name=name,
        )
        relationships = await rows(
            """MATCH (:Ontology {name:$name})-[:CONTAINS]->(r:RelationshipType)
            OPTIONAL MATCH (r)-[:FROM]->(a:NodeType)
            OPTIONAL MATCH (r)-[:TO]->(b:NodeType)
            WITH r, collect(DISTINCT a.name) AS sources, collect(DISTINCT b.name) AS targets
            OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:PropertyDef)
            OPTIONAL MATCH (p)-[:USES_ALIAS_MAP]->(am:AliasMap)
            OPTIONAL MATCH (p)-[:USES_BLOCKLIST]->(bl:Blocklist)
            RETURN properties(r) AS metadata, sources, targets,
                   collect(p{.*, alias_map_name:am.name, blocklist_name:bl.name}) AS properties ORDER BY metadata.name""",
            name=name,
        )
        aliases = await rows(
            """MATCH (:Ontology {name:$name})-[:DEFINES]->(a:AliasMap)
            OPTIONAL MATCH (a)-[:HAS_ALIAS]->(v:Alias)
            RETURN a.name AS name, collect(v{.from,.to}) AS entries""",
            name=name,
        )
        blocks = await rows(
            """MATCH (:Ontology {name:$name})-[:DEFINES]->(b:Blocklist)
            OPTIONAL MATCH (b)-[:HAS_TERM]->(v:BlockedTerm)
            RETURN b.name AS name, collect(v.value) AS terms""",
            name=name,
        )
        return metadata, nodes, relationships, aliases, blocks

    async with driver.session(database=database, default_access_mode="READ") as session:
        metadata, nodes, relationships, aliases, blocks = await session.execute_read(
            read
        )
    document = {
        "schema": SCHEMA_URI,
        "version": str(metadata.get("version") or "1"),
        "id": metadata["name"],
        "nodes": {},
        "relationships": {},
    }
    for key, source in [("name", "displayName"), ("description", "description")]:
        if metadata.get(source):
            document[key] = metadata[source]
    for row in nodes:
        meta = row["metadata"]
        name = meta["name"]
        if name in document["nodes"]:
            raise ValueError(f"Duplicate Ontology DB node type: {name}")
        document["nodes"][name] = {
            "label": name,
            "description": meta.get("description") or "",
            "properties": _db_properties(row["properties"]),
        }
    for row in relationships:
        meta = row["metadata"]
        name = meta["name"]
        if (
            name in document["relationships"]
            or len(row["sources"]) != 1
            or len(row["targets"]) != 1
        ):
            raise ValueError(
                f"Ontology DB relationship {name!r} requires one source and target and a unique name"
            )
        document["relationships"][name] = {
            "type": name,
            "description": meta.get("description") or "",
            "from": {"node": row["sources"][0]},
            "to": {"node": row["targets"][0]},
            "properties": _db_properties(row["properties"]),
        }
    custom = {
        "alias_maps": {
            row["name"]: {entry["from"]: entry["to"] for entry in row["entries"]}
            for row in aliases
        },
        "blocklists": {row["name"]: row["terms"] for row in blocks},
    }
    if aliases or blocks:
        document["extensions"] = {"custom": {TEMPLATE_EXTENSION: custom}}
    return validate_ontology(document)


def _db_properties(properties):
    result = {}
    for prop in properties:
        if not prop or not prop.get("name"):
            raise ValueError("Ontology DB properties must have nonempty names")
        if prop["name"] in result:
            raise ValueError(f"Duplicate Ontology DB property: {prop['name']}")
        result[prop["name"]] = _property_document(prop)
    return result


def _identifiers(identifier, value, relationship=False):
    return [
        identifier,
        *([value["type"]] if relationship else _node_labels(value)),
        *value.get("aliases", []),
    ]


def select_ontology(document, entities=None, search=None):
    """Match identifiers/aliases; preserve connecting relationships and their endpoints."""
    wanted = {name.casefold() for name in entities or []}
    needle = search.casefold() if search else None

    def matches(identifier, value, relationship=False):
        names = _identifiers(identifier, value, relationship)
        return (not wanted or bool(wanted & {name.casefold() for name in names})) and (
            not needle
            or any(needle in name.casefold() for name in names)
            or needle in value.get("description", "").casefold()
        )

    all_nodes = document.get("nodes", {})
    nodes = {nid: node for nid, node in all_nodes.items() if matches(nid, node)}
    rels = {}
    for rid, rel in document.get("relationships", {}).items():
        sources = _endpoint_ids(rel["from"], all_nodes)
        targets = _endpoint_ids(rel["to"], all_nodes)
        if matches(rid, rel, True) or (
            sources and targets and sources <= nodes.keys() and targets <= nodes.keys()
        ):
            rels[rid] = rel
    for rel in rels.values():
        for end in ("from", "to"):
            for nid in _endpoint_ids(rel[end], all_nodes):
                nodes[nid] = all_nodes[nid]
    selected = copy.deepcopy(document)
    selected["nodes"] = copy.deepcopy(nodes)
    selected["relationships"] = copy.deepcopy(rels)
    return selected


def _schema_card(document, include_properties):
    lines = [f"# {document.get('name') or document['id']} ontology", _CONTEXT]
    if document.get("description"):
        lines.append(document["description"])
    for nid, node in document.get("nodes", {}).items():
        labels = node.get("labels", {})
        lines.append(
            f"Node {nid}: identifying_label={json.dumps(labels.get('identifier') or node.get('label'))}; "
            f"implied_labels={json.dumps(labels.get('implied', []))}; "
            f"optional_labels={json.dumps(labels.get('optional', []))}; {node.get('description', '')}"
        )
        if node.get("label") and labels.get("identifier"):
            lines.append(f"  label shorthand: {json.dumps(node['label'])}")
        if node.get("aliases"):
            lines.append(f"  semantic aliases: {json.dumps(node['aliases'])}")
        if include_properties:
            for pid, prop in node.get("properties", {}).items():
                lines.append(
                    f"  property {prop.get('name') or pid}: {prop.get('type', 'ANY')}; {prop.get('description', '')}"
                )
    for rid, rel in document.get("relationships", {}).items():
        lines.append(
            f"Relationship {rid}: {rel['type']}, from={json.dumps(rel['from'])}, to={json.dumps(rel['to'])}; {rel.get('description', '')}"
        )
        if include_properties:
            for pid, prop in rel.get("properties", {}).items():
                lines.append(
                    f"  property {prop.get('name') or pid}: {prop.get('type', 'ANY')}; {prop.get('description', '')}"
                )
    lines.append(
        "The ontology object is authoritative for all declared constraints, allowed values, patterns, references and extensions."
    )
    return "\n".join(lines)


def ontology_response(
    document, source, entities=None, search=None, include_properties=True
):
    selected = select_ontology(document, entities, search)
    if not include_properties:
        for value in [*selected["nodes"].values(), *selected["relationships"].values()]:
            value.pop("properties", None)
    response = {
        "source": source,
        "ontology_id": document["id"],
        "ontology_version": document["version"],
        "context": _CONTEXT,
        "matched": bool(selected["nodes"] or selected["relationships"]),
        "mode": "ontology",
        "properties_included": include_properties,
        "scope": "filtered" if entities or search else "whole",
        "entities": [*_entity_index(selected)],
        "schema_card": _schema_card(selected, include_properties),
        "ontology": selected,
    }
    if not response["matched"]:
        return _index_response(document, response, no_match=True)
    if len(json.dumps(response, ensure_ascii=False)) <= RESPONSE_CHAR_BUDGET:
        return response
    return _index_response(selected, response)


def _index_response(document, response, no_match=False):
    # Keep a bounded list, never truncate a JSON document midway. Search can resolve
    # names outside the index, and exact entities requests narrow repeated calls.
    entries = _entity_index(document)
    result = {
        key: response[key]
        for key in ["source", "ontology_id", "ontology_version", "context", "matched"]
    }
    result.update(
        mode="index",
        entities=[],
        omitted_entities=len(entries),
        hint="Ontology exceeds response budget. Call again with entities=[...] or search='...' to narrow it; include_properties=false gives a structural overview.",
    )
    if no_match:
        result["hint"] = (
            "No ontology entity matched. Available entities follow; try id/label/alias or search, or physical schema discovery for ontology gaps."
        )
    for key in ["ontology_id", "ontology_version"]:
        if len(str(result[key])) > 200:
            del result[key]
            result["metadata_omitted"] = True
    for entry in entries:
        result["entities"].append(entry)
        result["omitted_entities"] -= 1
        if len(json.dumps(result, ensure_ascii=False)) > RESPONSE_CHAR_BUDGET:
            result["entities"].pop()
            result["omitted_entities"] += 1
            break
    return result


def _entity_index(document):
    return [
        *(
            {
                "id": nid,
                "kind": "node",
                "label": node.get("label") or node.get("labels", {}).get("identifier"),
                "aliases": node.get("aliases", []),
            }
            for nid, node in document.get("nodes", {}).items()
        ),
        *(
            {
                "id": rid,
                "kind": "relationship",
                "type": rel["type"],
                "aliases": rel.get("aliases", []),
            }
            for rid, rel in document.get("relationships", {}).items()
        ),
    ]


def register_ontology_tools(
    mcp,
    driver,
    database,
    namespace_prefix,
    read_timeout,
    *,
    ontology_path=None,
    ontology_driver=None,
    ontology_database=None,
    ontology_name=None,
):
    """Register read-only semantics retrieval without changing the physical schema tool."""
    configured_name = ontology_name

    @mcp.tool(
        name=namespace_prefix + "get_ontology",
        annotations=ToolAnnotations(
            title="Get domain ontology",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_ontology(
        ontology_name: Optional[str] = Field(
            default=None,
            description="Ontology DB name; omit to use the configured default or the sole available ontology. Cannot select files.",
        ),
        entities: Optional[list[str]] = Field(
            default=None,
            description="Optional node/relationship ids, database labels/types or semantic aliases; exact case-insensitive matches.",
        ),
        search: Optional[str] = Field(
            default=None,
            description="Optional substring in entity identifiers, aliases or descriptions. Combined with entities using AND.",
        ),
        include_properties: bool = True,
    ) -> dict[str, Any]:
        """Call FIRST for domain meaning when generating Cypher: descriptions, aliases,
        types, constraints, cardinality and allowed values from an ontology.

        Uses a deployment-configured spec file, otherwise the existing template
        Ontology DB. Does not sample user data or call a model. Declared semantics
        do not establish actual labels/indexes/population; use physical schema and
        read tools for those facts. A large result returns an index: narrow entities
        or search and call again. No match lists available ontology entities.
        """
        try:
            selected_name = ontology_name or configured_name
            if ontology_path:
                document = await asyncio.to_thread(load_ontology_file, ontology_path)
                if selected_name and selected_name != document["id"]:
                    raise ValueError(
                        f"Configured ontology file has id {document['id']!r}; requested {selected_name!r}"
                    )
                source = "file"
            else:
                document = await load_ontology_db(
                    ontology_driver if ontology_driver is not None else driver,
                    ontology_database or database,
                    selected_name,
                    read_timeout,
                )
                source = "template_ontology_db"
            response = ontology_response(
                document, source, entities, search, include_properties
            )
            return response
        except Exception as exc:
            raise ToolError(
                f"Ontology retrieval failed: {exc}. Physical schema discovery remains available via get_neo4j_schema_and_indexes."
            ) from exc
