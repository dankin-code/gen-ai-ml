# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Document Retrieval
# - Augmenting LLMs with the Expert Information (CONTEXT)... i.e. Creating Domain Expert Agent from a Generic LLM Model

# LIBRARIES 

# LLM Model API
from langchain_openai import ChatOpenAI

# VectorStores & Text Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# RAG 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Common
import yaml
import os
from dotenv import load_dotenv

from pprint import pprint
from IPython.display import Markdown


load_dotenv()


# Key Parameters
VECTOR_DATABASE = "./02 RAG Web Scraping/data/products_vectorstore.db"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL       = "gpt-4o-mini"

# OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# * 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE 

# Initialize the embedding function
embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

# Connect to the vector store
vectorstore = Chroma(
    persist_directory=VECTOR_DATABASE,
    embedding_function=embedding_function
)

# Create a Retriever
retriever = vectorstore.as_retriever()

retriever

# How the retriever works
retriever.invoke("What is in the 5 Course R Track?")


# * 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

prompt

# * Generic LLM Specification

model = ChatOpenAI(
    model = LLM_MODEL,
    temperature = 0.7,
)

model

model.invoke("What is the color of the sky? Answer in 1 word.")

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain

# * Try it out:

# * Baseline
result_baseline = model.invoke("What is inside the 5-Course R Track and what is the price? Cite your sources and URL.")

Markdown(result_baseline.content)

# * RAG
result = rag_chain.invoke("What is inside the 5-Course R Track and what is the price? Cite your sources and URL.")

Markdown(result)

# CONCLUSION:
# - The RAG (Retrieval-Augmented Generation) approach significantly improves the LLM's ability to answer questions by providing it with relevant context from the vector store. This allows for more accurate and context-aware responses.
# - This is one way to build domain-specific knowledge into a generative model, enhancing its performance on specialized tasks.
# - IMPORTANT: Many companies need this.
