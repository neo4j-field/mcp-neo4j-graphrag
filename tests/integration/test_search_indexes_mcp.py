"""Execute public ranked search on both legacy and SEARCH-capable databases."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
from fastmcp import Client
from neo4j import AsyncGraphDatabase
from mcp_neo4j_graphrag.server import create_mcp_server
from mcp_neo4j_graphrag.search_indexes import supports_search


@unittest.skipUnless(os.getenv('NEO4J_MCP_RUN_INTEGRATION')=='1','requires disposable Neo4j')
class SearchIndexesIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.driver=AsyncGraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.environ['NEO4J_USERNAME'],os.environ['NEO4J_PASSWORD']))
        self.database=os.getenv('NEO4J_DATABASE','neo4j')
        async with self.driver.session(database=self.database) as s:
            version = (await (await s.run('CALL dbms.components() YIELD versions RETURN versions[0] AS version')).single())['version']
        filters = ' WITH [n.group]' if supports_search(version,'VECTOR') else ''
        self.label='Search'+uuid4().hex
        self.vector=self.label+'Vector';self.fulltext=self.label+'Text';self.relationship=self.label+'Relationship'
        async with self.driver.session(database=self.database) as s:
            for query in [f'CREATE (n:{self.label} {{text:"coolant valve", vector:[1.0,0.0,0.0], group:"a"}})-[:{self.label} {{text:"coolant valve link"}}]->(:{self.label} {{text:"other", vector:[0.0,1.0,0.0], group:"b"}})',
                          f'CREATE VECTOR INDEX {self.vector} FOR (n:{self.label}) ON n.vector{filters} OPTIONS {{indexConfig: {{`vector.dimensions`:3,`vector.similarity_function`:"cosine"}}}}',
                          f'CREATE FULLTEXT INDEX {self.fulltext} FOR (n:{self.label}) ON EACH [n.text]',
                          f'CREATE FULLTEXT INDEX {self.relationship} FOR ()-[r:{self.label}]-() ON EACH [r.text]',
                          'CALL db.awaitIndexes(60)']:
                await (await s.run(query)).consume()
        self.client=Client(create_mcp_server(self.driver,'text-embedding-3-small',database=self.database))
        await self.client.__aenter__()

    async def asyncTearDown(self):
        await self.client.__aexit__(None,None,None)
        async with self.driver.session(database=self.database) as s:
            await (await s.run(f'MATCH (n:{self.label}) DETACH DELETE n')).consume()
            for name in [self.vector,self.fulltext,self.relationship]:
                await (await s.run(f'DROP INDEX {name} IF EXISTS')).consume()
        await self.driver.close()

    async def test_public_vector_and_node_relationship_fulltext(self):
        with patch('mcp_neo4j_graphrag.server.litellm.embedding',return_value=SimpleNamespace(data=[{'embedding':[1.,0.,0.]}])):
            response=await self.client.call_tool('vector_search',dict(text_query='coolant',vector_index=self.vector,top_k=1,pre_filter={'group':'a'},return_properties='text'))
        result=json.loads(response.content[0].text)['results']
        self.assertEqual('coolant valve',result[0]['text'])
        for index,projection in [(self.fulltext,'nodeId'),(self.relationship,'relationshipId')]:
            response=await self.client.call_tool('fulltext_search',dict(text_query='coolant',fulltext_index=index,return_properties='text'))
            result=json.loads(response.content[0].text)['results']
            self.assertEqual(1,len(result));self.assertIn(projection,result[0]);self.assertIn('coolant',result[0]['text'])
            response=await self.client.call_tool('fulltext_search',dict(text_query='coolant',fulltext_index=index))
            self.assertIn('coolant',json.loads(response.content[0].text)['results'][0]['properties']['text'])
