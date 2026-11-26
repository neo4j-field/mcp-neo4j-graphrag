"""Version-aware ranked search clauses; provider/auth/query failures never trigger fallback."""
import re
from neo4j import RoutingControl


def supports_search(version: str, kind: str) -> bool:
    match = re.match(r'^(\d+)\.(\d+)', version)
    if not match:
        raise ValueError(f'Cannot determine search syntax for Neo4j version {version!r}')
    release = tuple(map(int, match.groups()))
    return release >= ((2026, 1) if kind == 'VECTOR' else (2026, 9))


async def search_source(driver, database, kind, index_name, *, relationship=False, filter_predicate=None):
    rows = await driver.execute_query(
        'CALL dbms.components() YIELD versions RETURN versions[0] AS version',
        routing_control=RoutingControl.READ, database_=database,
        result_transformer_=lambda r: r.data(),
    )
    if not rows:
        raise RuntimeError('Neo4j did not return its version; cannot select search syntax')
    quoted_index = '`' + index_name.replace('`', '``') + '`'
    variable = 'relationship' if relationship else 'node'
    if supports_search(rows[0]['version'], kind):
        pattern = '()-[relationship]->()' if relationship else '(node)'
        query = '$query_vector' if kind == 'VECTOR' else '$query'
        limit = '$fetch_k' if kind == 'VECTOR' else '$top_k'
        filters = f' WHERE {filter_predicate}' if kind == 'VECTOR' and filter_predicate else ''
        return (f'CYPHER 25 MATCH {pattern} SEARCH {variable} IN '
                f'({kind} INDEX {quoted_index} FOR {query}{filters} LIMIT {limit}) SCORE AS score')
    procedure = 'vector' if kind == 'VECTOR' else 'fulltext'
    method = 'queryRelationships' if relationship else 'queryNodes'
    args = '$index_name, $fetch_k, $query_vector' if kind == 'VECTOR' else '$index_name, $query'
    return f'CALL db.index.{procedure}.{method}({args}) YIELD {variable}, score'
