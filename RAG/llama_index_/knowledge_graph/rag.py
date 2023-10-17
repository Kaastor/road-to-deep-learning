'''
https://betterprogramming.pub/7-query-strategies-for-navigating-knowledge-graphs-with-llamaindex-ed551863d416
https://colab.research.google.com/drive/1tLjOg2ZQuIClfuWrAC2LdiZHCov8oUbs?usp=sharing#scrollTo=iDjEGsguhCzw

Connect to nebula-console: ./nebula-console -addr 127.0.0.1 -port 9669 -u root -p nebula
In Nebula console execute:
CREATE SPACE IF NOT EXISTS phillies_rag(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
USE phillies_rag;
CREATE TAG IF NOT EXISTS entity(name string);
CREATE EDGE IF NOT EXISTS relationship(relationship string);
CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));
'''
import os
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index import (
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index import download_loader
from llama_index.llms import OpenAI

os.environ["GRAPHD_HOST"] = "127.0.0.1"
os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ["NEBULA_ADDRESS"] = "127.0.0.1:9669"

connection_string = f"--address {os.environ['GRAPHD_HOST']} --port 9669 --user root --password {os.environ['NEBULA_PASSWORD']}"

space_name = "phillies_rag"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# define LLM
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

from llama_index import load_index_from_storage
from llama_hub.youtube_transcript import YoutubeTranscriptReader

kg_index = None
# check in our local storage_context to load the KG index
try:
    storage_context = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
    kg_index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        max_triplets_per_chunk=15,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        verbose=True,
    )
    index_loaded = True
except:
    index_loaded = False

# we need to load the two source documents, from which we build the KG index
# and then persist the doc store, index store, and vector store in the local
# storage_graph directory at the project root
if not index_loaded:
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    wiki_documents = loader.load_data(pages=['Philadelphia Phillies'], auto_suggest=False)
    print(f'Loaded {len(wiki_documents)} documents')

    youtube_loader = YoutubeTranscriptReader()
    youtube_documents = youtube_loader.load_data(ytlinks=['https://www.youtube.com/watch?v=k-HTQ8T7oVw'])
    print(f'Loaded {len(youtube_documents)} YouTube documents')

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=wiki_documents + youtube_documents,
        storage_context=storage_context,
        max_triplets_per_chunk=15,
        service_context=service_context,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        # This can be useful when you want to perform a semantic search on the
        # Knowledge Graph, as the embeddings can be used to find nodes and edges
        # that are semantically similar to the query.
        include_embeddings=True,
    )

    kg_index.storage_context.persist(persist_dir='./storage_graph')


query_engine = kg_index.as_query_engine()
response = query_engine.query("Tell me about some of the facts of Philadelphia Phillies.")
print(f"<b>{response}</b>")
