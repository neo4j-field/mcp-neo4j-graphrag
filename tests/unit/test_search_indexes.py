import unittest
from unittest.mock import AsyncMock, MagicMock
from mcp_neo4j_graphrag.search_indexes import supports_search, search_source


class SearchSyntaxTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_boundaries_and_procedure_compatibility(self):
        for version, vector, fulltext in [('5.26.0',False,False),('2025.12.1',False,False),('2026.01.0',True,False),('2026.08.0',True,False),('2026.09.0',True,True)]:
            self.assertEqual(vector,supports_search(version,'VECTOR'))
            self.assertEqual(fulltext,supports_search(version,'FULLTEXT'))
            driver=MagicMock(execute_query=AsyncMock(return_value=[{'version':version}]))
            for kind,modern in [('VECTOR',vector),('FULLTEXT',fulltext)]:
                for relationship in [False,True]:
                    source=await search_source(driver,'neo4j',kind,'index`name',relationship=relationship)
                    self.assertEqual(modern,source.startswith('CYPHER 25'))
                    self.assertIn('relationship' if relationship else 'node',source.lower())
                    self.assertIn('SCORE AS score' if modern else 'YIELD',source)

    async def test_lookup_errors_do_not_fallback(self):
        driver=MagicMock(execute_query=AsyncMock(side_effect=RuntimeError('permission denied')))
        with self.assertRaisesRegex(RuntimeError,'permission denied'):
            await search_source(driver,'neo4j','VECTOR','index')
        with self.assertRaises(ValueError):
            supports_search('unknown','VECTOR')

    async def test_public_search_refuses_non_online_before_embedding(self):
        from fastmcp import Client
        from unittest.mock import patch
        from mcp_neo4j_graphrag.server import create_mcp_server
        for state in ['POPULATING', 'FAILED']:
            driver=MagicMock(execute_query=AsyncMock(return_value=[dict(properties=['vector'],entityType='NODE',state=state)]))
            with patch('mcp_neo4j_graphrag.server.litellm.embedding') as provider:
                async with Client(create_mcp_server(driver,'text-embedding-3-small')) as client:
                    for tool,args in [('vector_search',dict(text_query='test',vector_index='index')),('fulltext_search',dict(text_query='test',fulltext_index='index'))]:
                        result=await client.call_tool(tool,args,raise_on_error=False)
                        self.assertTrue(result.is_error)
                        self.assertIn(state,str(result))
                provider.assert_not_called()
