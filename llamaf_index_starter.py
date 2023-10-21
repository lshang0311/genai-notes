from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import set_global_service_context

from llm_initialization import init_llm, init_embeddings
from utils import load_env

"""
Ref:
    https://gpt-index.readthedocs.io/en/stable/getting_started/starter_example.html
"""

# documents
documents = SimpleDirectoryReader('data').load_data()

# llm
env = "TEST"
load_env(env)
llm = init_llm()

embed_model = init_embeddings()

# vector store
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)
index = VectorStoreIndex.from_documents(documents)

# query
query = "what were the two things that the author was working on before college?"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)
