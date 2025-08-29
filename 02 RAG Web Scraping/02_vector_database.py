# RETRIEVAL-AUGMENTED GENERATION (RAG)

# How to inject our LLM's with knowledge: RAG Part 1

# Goals: Intro to ... 
# - Text Embeddings
# - Vector Databases

# LIBRARIES 

# Vector Store and Text Embeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Common
import re
import copy
import joblib

from pprint import pprint
import os
from dotenv import load_dotenv


load_dotenv()

# OPENAI API SETUP
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Resource: https://platform.openai.com/docs/models/text-embedding-ada-002
MODEL_EMBEDDING = 'text-embedding-ada-002' # OpenAI Text Embedding Model that works the best in my experience

# * 1.0 DATA PREPARATION ----

# Load the documents
documents = joblib.load("./02 RAG Web Scraping/data/products.pkl")

documents

for doc in documents:
    print(len(doc.page_content))
    
# * Preprocessing Text - Cleaning the page_content

def clean_text(text):

    text = re.sub(r'\n+', '\n', text) 
    text = re.sub(r'\s+', ' ', text)  

    text = re.sub(r'Toggle navigation.*?Business Science', '', text, flags=re.DOTALL)
    text = re.sub(r'© Business Science University.*', '', text, flags=re.DOTALL)

    # Replace encoded characters
    text = text.replace('\xa0', ' ')
    text = text.replace('ðŸŽ‰', '')  

    # Extract relevant content
    relevant_content = []
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ["Enroll in Course", "data scientist", "promotion", "salary", "testimonial"]):
            relevant_content.append(line.strip())

    # Join the relevant content back into a single string
    cleaned_text = '\n'.join(relevant_content)

    return cleaned_text


# Test cleaning a single document

pprint(documents[0].page_content)

pprint(clean_text(documents[0].page_content))

# Clean all documents

documents_clean = copy.deepcopy(documents)

for document in documents_clean:
    document.page_content = clean_text(document.page_content)
    
documents_clean

len(documents_clean)

pprint(documents_clean[0].page_content)

# * Pre Processing Text - Adding Metadata to page content
# - Helps with RAG by adding sources and searching titles

documents_clean[0].metadata

documents_with_metadata = copy.deepcopy(documents_clean)

for doc in documents_with_metadata:
    # Retrieve the title and source from the document's metadata
    title = doc.metadata.get('title', 'Unknown Title')
    source = doc.metadata.get('source', 'Unknown URL')
    
    # Prepend the title and source to the page content
    updated_content = f"Title: {title} \nSource: {source} \n\n {doc.page_content}"

    # Update the document's page content
    doc.page_content = updated_content

pprint(documents_with_metadata[0].page_content)

# * Additional pre-processing can be done here if needed with AI
#   - For example, we can use and AI model to summarize this text extracting key features. 
#   - This is called synthetic data generation, and you will learn how in Clinic 6


# * 2.0 VECTOR STORES

# * Text Embedding Models

# OpenAI Text Embedding Models
# - See available models: https://platform.openai.com/docs/models
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview


# NOTE: The embedding model selected needs to be used every time you access the Vector Store
embedding_function = OpenAIEmbeddings(model=MODEL_EMBEDDING,)

embedding_function


# ** Vector Store - Complete (Large) Documents

# Create the Vector Store (Run 1st Time)
# Chroma.from_documents(
#     documents_with_metadata, 
#     embedding=embedding_function, 
#     persist_directory="data/products_vectorstore_2.db"
# )

# Connect to the Vector Store (Run all other times)
vectorstore = Chroma(
    embedding_function=embedding_function, 
    persist_directory="./02 RAG Web Scraping/data/products_vectorstore.db"
)

vectorstore


# * Similarity Search: The whole reason we did this

result = vectorstore.similarity_search(
    query="What is 5 Course R Track?", 
    k = 5
)

result

pprint(result[0].page_content)

# CONCLUSION:
# - Went from raw text documents to a structured vector database
# - Enabled efficient similarity search for relevant information