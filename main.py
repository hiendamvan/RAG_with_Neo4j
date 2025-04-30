import os
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY=os.getenv("COHERE_API_KEY")
#create llm using Cohere 
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r-plus"
)

#create embedding 
embedding_provider = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model='embed-english-v3.0'
)

# connect to graph database 

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph, 
    index_name='moviePlots',
    embedding_node_property='plotEmbedding',
    text_node_property='plot'
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(response)


