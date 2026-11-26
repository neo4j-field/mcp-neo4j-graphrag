"""Bounded, model-free navigation of the workspace's existing lexical graph."""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from weakref import WeakKeyDictionary

import anyio

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from neo4j.exceptions import DriverError, Neo4jError
from mcp.types import ToolAnnotations
from neo4j import Query, READ_ACCESS
from pydantic import Field


logger = logging.getLogger(__name__)

ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
DOC = "MATCH (d:Document {id:$document_id}) WHERE coalesce(d.active,true) = true\n"
# HAS_SECTION connects a Document to ALL its sections, not just roots.
OUTLINE = (
    DOC
    + """
MATCH (d)-[:HAS_SECTION]->(top:Section)
WHERE NOT EXISTS { MATCH (d)-[:HAS_SECTION]->(parent:Section)-[:HAS_SUBSECTION]->(top) }
MATCH path=(top)-[:HAS_SUBSECTION*0..32]->(s:Section)
WHERE all(n IN nodes(path) WHERE EXISTS { MATCH (d)-[:HAS_SECTION]->(n) })
WITH d,s,path, CASE WHEN $root_section_id IS NULL THEN 0
     ELSE [i IN range(0,length(path)) WHERE nodes(path)[i].id=$root_section_id][0] END AS root_index
WHERE root_index IS NOT NULL AND length(path)-root_index < $depth
RETURN DISTINCT s.id AS section_id, s.title AS title, s.level AS level,
       length(path)-root_index AS depth,
       CASE WHEN length(path)=0 THEN null ELSE nodes(path)[-2].id END AS parent_section_id,
       [n IN nodes(path) | {section_id:n.id,title:n.title}] AS path,
       s.order AS section_order, s.pageStart AS page_start_index,s.pageEnd AS page_end_index,
       EXISTS { MATCH (s)-[:HAS_SUBSECTION]->(child:Section)<-[:HAS_SECTION]-(d) } AS has_children
ORDER BY coalesce(section_order,toInteger(split(section_id,'_sec_')[-1]),2147483647), section_id
SKIP $offset LIMIT $take
"""
)
SECTION_ELEMENTS = (
    DOC
    + """
MATCH (d)-[:HAS_SECTION]->(root:Section {id:$section_id})
MATCH path=(root)-[:HAS_SUBSECTION*0..32]->(s:Section)
WHERE ($include_descendants OR length(path)=0)
  AND all(n IN nodes(path) WHERE EXISTS { MATCH (d)-[:HAS_SECTION]->(n) })
MATCH (s)-[:HAS_ELEMENT]->(e:Element)
WITH DISTINCT e
"""
)
DOCUMENT_ELEMENTS = (
    DOC
    + """
CALL {
 WITH d
 MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_ELEMENT]->(e:Element) RETURN e
 UNION
 WITH d
 MATCH (d)-[:HAS_ELEMENT]->(e:Element) RETURN e
 UNION
 WITH d
 MATCH (d)-[:HAS_SECTION]->(:Section)-[:HAS_ELEMENT]->(e:Element) RETURN e
 UNION
 WITH d
 MATCH (e:Chunk)-[:PART_OF]->(d)
 WHERE coalesce(e.active,true)=true AND coalesce(d.parseMode,'') <> 'page_image'
   AND NOT EXISTS { MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_ELEMENT]->(:Element) }
   AND NOT EXISTS { MATCH (d)-[:HAS_ELEMENT]->(:Element) }
   AND NOT EXISTS { MATCH (d)-[:HAS_SECTION]->(:Section)-[:HAS_ELEMENT]->(:Element) }
 RETURN e
 UNION
 WITH d
 MATCH (d)-[:HAS_PAGE]->(e:Page)
 WHERE NOT EXISTS { MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_ELEMENT]->(:Element) }
   AND NOT EXISTS { MATCH (d)-[:HAS_ELEMENT]->(:Element) }
   AND NOT EXISTS { MATCH (d)-[:HAS_SECTION]->(:Section)-[:HAS_ELEMENT]->(:Element) }
   AND (d.parseMode='page_image' OR NOT EXISTS { MATCH (c:Chunk)-[:PART_OF]->(d) WHERE coalesce(c.active,true)=true })
 RETURN e
}
WITH DISTINCT e
"""
)
CONTENT_TAIL = """
WITH e ORDER BY coalesce(e.pageNumber,2147483647),
 coalesce(e.order,toInteger(split(e.id,'_elem_')[-1]),e.index,2147483647),e.id
SKIP $offset LIMIT $take
WITH collect(e) AS items
UNWIND range(0,size(items)-1) AS i
WITH items[i] AS e,i, CASE WHEN i=0 THEN $text_offset ELSE 0 END AS start
RETURN e.id AS element_id, elementId(e) AS node_element_id, labels(e) AS labels, e.type AS type,
 e.pageNumber AS page_number, e.sourceAnchor AS source_anchor,
 [(e)<-[:HAS_ELEMENT]-(section:Section)<-[:HAS_SECTION]-(owner:Document {id:$document_id}) | section.id] AS section_ids,
 substring(CASE WHEN coalesce(e.text,'') <> '' THEN e.text ELSE coalesce(e.textDescription,'') END, start, $max_chars) AS text,
 size(CASE WHEN coalesce(e.text,'') <> '' THEN e.text ELSE coalesce(e.textDescription,'') END) AS text_length,
 start AS text_offset,
 EXISTS { MATCH (e)<-[:HAS_ELEMENT]-(:Page) } AS physical_page_link
ORDER BY i
"""


CAPABILITIES = """
RETURN EXISTS {
 MATCH (d:Document) WHERE d.id IS NOT NULL AND coalesce(d.active,true)=true AND (
   EXISTS { MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_ELEMENT]->(e:Element) WHERE e.id IS NOT NULL AND (coalesce(e.text,'') <> '' OR coalesce(e.textDescription,'') <> '' OR coalesce(e.imageBase64,'') <> '') }
   OR EXISTS { MATCH (d)-[:HAS_ELEMENT]->(e:Element) WHERE e.id IS NOT NULL AND (coalesce(e.text,'') <> '' OR coalesce(e.textDescription,'') <> '' OR coalesce(e.imageBase64,'') <> '') }
   OR EXISTS { MATCH (d)-[:HAS_SECTION]->(:Section)-[:HAS_ELEMENT]->(e:Element) WHERE e.id IS NOT NULL AND (coalesce(e.text,'') <> '' OR coalesce(e.textDescription,'') <> '' OR coalesce(e.imageBase64,'') <> '') }
   OR EXISTS { MATCH (c:Chunk)-[:PART_OF]->(d)
               WHERE c.id IS NOT NULL AND coalesce(c.active,true)=true
                 AND (coalesce(c.text,'') <> '' OR coalesce(c.textDescription,'') <> '') }
   OR EXISTS { MATCH (d)-[:HAS_PAGE]->(p:Page)
               WHERE p.id IS NOT NULL AND (coalesce(p.text,'') <> '' OR coalesce(p.textDescription,'') <> '')
                 AND (d.parseMode='page_image' OR NOT EXISTS {
                   MATCH (c:Chunk)-[:PART_OF]->(d) WHERE coalesce(c.active,true)=true }) }
 )
} AS documents,
EXISTS {
 MATCH (d:Document)-[:HAS_SECTION]->(s:Section)-[:HAS_ELEMENT]->(e:Element)
 WHERE d.id IS NOT NULL AND s.id IS NOT NULL AND e.id IS NOT NULL
   AND (coalesce(e.text,'') <> '' OR coalesce(e.textDescription,'') <> '' OR coalesce(e.imageBase64,'') <> '')
   AND coalesce(d.active,true)=true
} AS sections
"""


class _PageIndexDiscovery(Middleware):
    """Hide unsupported tools and notify connected clients after external ingestion.

    The server lifespan owns one observer, with one capability query per tick
    shared by sessions. Clients must honor MCP tools/list_changed notifications.
    """

    def __init__(self, driver, database, prefix, timeout, poll_interval=2.0):
        self.driver, self.database, self.timeout = driver, database, timeout
        self.document_tools = {
            prefix + name
            for name in ("list_retrieval_documents", "get_document_content")
        }
        self.section_tools = {
            prefix + name for name in ("get_document_outline", "get_section_content")
        }
        self.navigation_tools = self.document_tools | self.section_tools
        self.poll_interval = poll_interval
        # Each client caches its own tools/list result. Do not use a global
        # previous state: a newly connected client must not consume a change
        # notification owed to an older client.
        self._sessions = WeakKeyDictionary()

    @asynccontextmanager
    async def lifespan(self, server):
        """Observe external ingestion only while this MCP server is running."""
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(self._observe)
            try:
                yield {}
            finally:
                tasks.cancel_scope.cancel()
                self._sessions.clear()

    async def _observe(self):
        while True:
            await anyio.sleep(self.poll_interval)
            if self._sessions:
                try:
                    await self._refresh_sessions()
                except Exception:
                    # A background discovery failure must not terminate the MCP
                    # lifespan. Cancellation still propagates (BaseException).
                    logger.exception("Lexical tool availability refresh failed; retaining client state")

    async def _refresh_sessions(self):
        # A failed probe is not evidence that lexical content disappeared.
        # Let the observer retain cached capabilities and retry on its next tick.
        enabled = await self.enabled(hide_on_error=False)
        for session, previous in list(self._sessions.items()):
            if previous == enabled:
                continue
            # Store before sending: a client may immediately re-list on receipt.
            self._sessions[session] = enabled
            try:
                await session.send_tool_list_changed()
            except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
                self._sessions.pop(session, None)
            except Exception:
                # A broken client notification channel must not affect other
                # sessions. A subsequent tools/list re-enrolls this session.
                self._sessions.pop(session, None)
                logger.exception("Cannot notify MCP client of changed lexical tools")

    async def enabled(self, *, hide_on_error=True):
        try:
            rows = await _query(self.driver, self.database, self.timeout, CAPABILITIES)
        except (Neo4jError, DriverError) as exc:
            if not hide_on_error:
                raise
            logger.warning(
                "Cannot inspect lexical graph tool availability; hiding navigation tools: %s",
                exc,
            )
            return set()
        if len(rows) != 1 or not all(
            key in rows[0] for key in ("documents", "sections")
        ):
            logger.warning(
                "Lexical graph availability query returned an unexpected result; hiding navigation tools"
            )
            return set()
        enabled = set()
        if rows[0]["documents"]:
            enabled.update(self.document_tools)
        if rows[0]["sections"]:
            enabled.update(self.section_tools)
        return enabled

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        enabled = await self.enabled()
        if context.fastmcp_context is not None:
            self._sessions[context.fastmcp_context.session] = enabled
        return [
            tool
            for tool in tools
            if tool.name not in self.navigation_tools or tool.name in enabled
        ]

    async def on_call_tool(self, context, call_next):
        name = context.message.name
        if name in self.navigation_tools and name not in await self.enabled():
            raise ToolError(
                "This navigation tool is unavailable: supported active document/section topology "
                "could not be established. Check database connectivity/read permissions or ingest "
                "lexical data, then list tools again."
            )
        return await call_next(context)


def _bounds(offset, limit, maximum, text_offset=0, max_chars=12000):
    if not 0 <= offset <= 1_000_000 or not 1 <= limit <= maximum:
        raise ToolError(f"offset must be 0..1000000 and limit 1..{maximum}")
    if not 0 <= text_offset <= 10_000_000 or not 1 <= max_chars <= 20000:
        raise ToolError("text_offset must be 0..10000000; max_chars 1..20000")


def _page(rows, offset, limit):
    more = len(rows) > limit
    return {
        "items": rows[:limit],
        "offset": offset,
        "next_offset": offset + limit if more else None,
        "has_more": more,
    }


async def _query(driver, database, timeout, statement, **parameters):
    session_options = {}
    if statement == CAPABILITIES:
        # Missing lexical labels/properties are expected on an empty or business-
        # only graph. Suppress probe notifications, not retrieval diagnostics/errors.
        session_options["notifications_min_severity"] = "OFF"
    async with driver.session(
        database=database, default_access_mode=READ_ACCESS, **session_options
    ) as session:
        result = await session.run(
            Query(statement, timeout=timeout), parameters=parameters
        )
        return await result.data()


async def _document(driver, database, timeout, document_id):
    rows = await _query(
        driver,
        database,
        timeout,
        DOC
        + """
        RETURN d.id AS document_id,d.name AS document_name,d.source AS source,
               d.sourceFormat AS source_format,
               d.sourceId AS source_id,d.version AS version,d.parseMode AS parse_mode,
               COUNT { (d)-[:HAS_SECTION]->(:Section) } AS section_count LIMIT 2
    """,
        document_id=document_id,
    )
    if len(rows) != 1:
        raise ToolError("Active document ID must resolve to exactly one document")
    return rows[0]


def _content_page(rows, document, offset, text_offset, limit, max_chars):
    items, remaining, continuation = [], max_chars, None
    for index, row in enumerate(rows[:limit]):
        length, start = row["text_length"], row["text_offset"]
        if start > length:
            raise ToolError("text_offset exceeds this element's text length")
        text = row["text"][:remaining]
        consumed = len(text)
        page_index = row.pop("page_number", None)
        if document.get("source_format") in {"markdown", "text", "html", "docx"}:
            page_index = None
        item = {
            **row,
            "text": text,
            "text_end": start + consumed,
            "citation": {
                **document,
                "element_id": row["element_id"],
                "source_anchor": row.get("source_anchor"),
                "page_index": page_index,
                "section_ids": row.get("section_ids", []),
                "page_number": page_index + 1 if page_index is not None else None,
            },
        }
        item.pop("physical_page_link", None)
        items.append(item)
        remaining -= consumed
        if start + consumed < length:
            continuation = {"offset": offset + index, "text_offset": start + consumed}
            break
        if remaining == 0 or index + 1 == limit:
            if index + 1 < len(rows):
                continuation = {"offset": offset + index + 1, "text_offset": 0}
            break
    return {
        "document": document,
        "items": items,
        "next": continuation,
        "has_more": continuation is not None,
        "offset": offset,
        "text_offset": text_offset,
        "characters_returned": max_chars - remaining,
        "order": "page then original element order; offsets are stable while graph is unchanged",
        "page_number_convention": "citation.page_number is 1-based; page_index is stored 0-based; null means unavailable",
    }


def register_pageindex_tools(
    mcp, driver, database, namespace_prefix="", read_timeout=30, *, discovery=None
):
    mcp.add_middleware(
        discovery or _PageIndexDiscovery(driver, database, namespace_prefix, read_timeout)
    )

    @mcp.tool(
        name=namespace_prefix + "list_retrieval_documents", annotations=ANNOTATIONS
    )
    async def list_retrieval_documents(
        query: Optional[str] = None,
        offset: int = Field(default=0, ge=0, le=1_000_000),
        limit: int = Field(default=20, ge=1, le=100),
    ) -> dict:
        """Discover active lexical documents by literal name/ID substring; paginate results."""
        _bounds(offset, limit, 100)
        if query is not None and len(query) > 500:
            raise ToolError("query must be at most 500 characters")
        rows = await _query(
            driver,
            database,
            read_timeout,
            """
            MATCH (d:Document) WHERE coalesce(d.active,true)=true
              AND ($query IS NULL OR toLower(coalesce(d.name,'')) CONTAINS toLower($query)
                   OR toLower(d.id) CONTAINS toLower($query))
            RETURN d.id AS document_id,d.name AS document_name,d.source AS source,
               d.sourceFormat AS source_format,
                   d.version AS version,d.parseMode AS parse_mode,d.totalPages AS source_page_count,
                   COUNT { (d)-[:HAS_SECTION]->(:Section) } AS section_count
            ORDER BY document_name,document_id SKIP $offset LIMIT $take
        """,
            query=query,
            offset=offset,
            take=limit + 1,
        )
        return _page(rows, offset, limit)

    @mcp.tool(name=namespace_prefix + "get_document_outline", annotations=ANNOTATIONS)
    async def get_document_outline(
        document_id: str,
        root_section_id: Optional[str] = None,
        depth: int = Field(default=3, ge=1, le=12),
        offset: int = Field(default=0, ge=0, le=1_000_000),
        limit: int = Field(default=50, ge=1, le=100),
    ) -> dict:
        """Read a document ToC, then drill into section IDs. Paths include ancestors.

        depth counts visible levels (1 returns roots only). No inferred summaries.
        Tree traversal is capped at 32 edges. has_children indicates drilldown;
        next_offset pages the current depth, not hidden descendants.
        """
        _bounds(offset, limit, 100)
        if not 1 <= depth <= 12:
            raise ToolError("depth must be 1..12")
        document = await _document(driver, database, read_timeout, document_id)
        if root_section_id is not None:
            await section_exists(document_id, root_section_id)
        rows = await _query(
            driver,
            database,
            read_timeout,
            OUTLINE,
            document_id=document_id,
            root_section_id=root_section_id,
            depth=depth,
            offset=offset,
            take=limit + 1,
        )
        return {
            "document": document,
            **_page(rows, offset, limit),
            "depth": depth,
            "traversal_depth_cap": 32,
            "structure_available": document.get("section_count", 0) > 0,
            "page_empty": not rows,
            "page_range_convention": "stored 0-based; null means unavailable on legacy parses",
            "content_tool": "get_document_content"
            if document.get("section_count", 0) == 0
            else "get_section_content",
        }

    async def section_exists(document_id, section_id):
        rows = await _query(
            driver,
            database,
            read_timeout,
            DOC
            + """
            MATCH (d)-[:HAS_SECTION]->(s:Section {id:$section_id}) RETURN s.id AS id LIMIT 2
        """,
            document_id=document_id,
            section_id=section_id,
        )
        if len(rows) != 1:
            raise ToolError(
                "Section ID must belong to exactly one section in this active document"
            )

    async def content(
        document_id,
        section_id,
        include_descendants,
        offset,
        text_offset,
        limit,
        max_chars,
    ):
        _bounds(offset, limit, 25, text_offset, max_chars)
        document = await _document(driver, database, read_timeout, document_id)
        if section_id is not None:
            await section_exists(document_id, section_id)
        rows = await _query(
            driver,
            database,
            read_timeout,
            (SECTION_ELEMENTS if section_id is not None else DOCUMENT_ELEMENTS)
            + CONTENT_TAIL,
            document_id=document_id,
            section_id=section_id,
            include_descendants=include_descendants,
            offset=offset,
            text_offset=text_offset,
            max_chars=max_chars,
            take=limit + 1,
        )
        if not rows and text_offset:
            raise ToolError("text_offset requires an existing element at offset")
        path_rows = []
        if section_id is not None:
            path_rows = await _query(
                driver,
                database,
                read_timeout,
                OUTLINE,
                document_id=document_id,
                root_section_id=section_id,
                depth=1,
                offset=0,
                take=1,
            )
        return {
            **_content_page(rows, document, offset, text_offset, limit, max_chars),
            "section_path": path_rows[0]["path"] if path_rows else None,
            "structure_available": document.get("section_count", 0) > 0,
            "fallback_used": any(
                "Element" not in row["labels"] for row in rows[:limit]
            ),
            "section_id": section_id,
            "include_descendants": include_descendants,
            "traversal_depth_cap": 32 if section_id is not None else None,
            "pagination_scope": "selected elements within depth cap; deeper descendants are excluded"
            if section_id is not None
            else "all document elements or fallback content",
            "content_source": "original elements; active chunks/pages only when document has no elements",
            "visual_content": "text/available descriptions only; use read_node_image for images",
        }

    @mcp.tool(name=namespace_prefix + "get_section_content", annotations=ANNOTATIONS)
    async def get_section_content(
        document_id: str,
        section_id: str,
        include_descendants: bool = True,
        offset: int = Field(default=0, ge=0, le=1_000_000),
        text_offset: int = Field(default=0, ge=0, le=10_000_000),
        limit: int = Field(default=20, ge=1, le=25),
        max_chars: int = Field(default=12000, ge=1, le=20000),
    ) -> dict:
        """Read original section elements with document/page citations, not partial chunks.

        Follow next.offset and next.text_offset exactly to read all returned text.
        Descendant traversal is capped at 32 edges; graph must remain unchanged
        during pagination. An empty visual element is not evidence of absence.
        """
        return await content(
            document_id,
            section_id,
            include_descendants,
            offset,
            text_offset,
            limit,
            max_chars,
        )

    @mcp.tool(name=namespace_prefix + "get_document_content", annotations=ANNOTATIONS)
    async def get_document_content(
        document_id: str,
        offset: int = Field(default=0, ge=0, le=1_000_000),
        text_offset: int = Field(default=0, ge=0, le=10_000_000),
        limit: int = Field(default=20, ge=1, le=25),
        max_chars: int = Field(default=12000, ge=1, le=20000),
    ) -> dict:
        """Read a heading-free document or front matter using the same resumable bounds.

        Uses original Elements; when none exist uses active Chunks (e.g. PyMuPDF),
        or Page text/descriptions for page-image documents or when no active chunks
        exist. Does not reconstruct images.
        """
        return await content(
            document_id, None, True, offset, text_offset, limit, max_chars
        )
