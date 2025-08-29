# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goal: Collect Expertise on a Topic that's relevant to your business application

# WEB SCRAPING SETUP 
# ------------------------------------------------
# 1. Install required packages if needed (langchain_community for WebBaseLoader, beautifulsoup4 for parsing)
# 2. Note: Web scraping should respect website terms, robots.txt, and legal guidelines. This is for educational purposes only.


# * 1.0 LIBRARIES

from langchain.document_loaders import WebBaseLoader

# Other Libraries
import pandas as pd
import joblib

import nest_asyncio # Used for asynchronously loading

from pprint import pprint


# * 2.0 WEB PAGE LOADING

# * Test out loading a single webpage
#   Resource: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html#langchain_community.document_loaders.web_base.WebBaseLoader

url = "https://university.business-science.io/p/4-course-bundle-machine-learning-and-web-applications-r-track-101-102-201-202a"

# Create a document loader for the website
loader = WebBaseLoader(url)

# Load the data from the website
documents = loader.load()

# Returns a list containing documents
documents

len(documents)

documents[0]

dict(documents[0]).keys()

pprint(documents[0].metadata)

pprint(documents[0].page_content)

# * Load All Webpages Asynchronously
#   This will take a minute

# Get the websites
df = pd.read_csv("./02 RAG Web Scraping/data/products.csv")

df['website']

# Create a loader
loader = WebBaseLoader(df['website'].tolist())

# Load Synchronously (Takes about 16 seconds for 12 webpages)
documents = loader.load()

# * NEW - Load Asynchronously (Takes about 2 seconds for 12 webpages)
nest_asyncio.apply()

documents = loader.aload()

# Examine documents

documents

documents[0].metadata

pprint(documents[0].page_content)

len(documents[0].page_content)

# * 3.0 SAVE DOCUMENTS

# joblib.dump(documents, "data/products.pkl")

documents = joblib.load("./02 RAG Web Scraping/data/products.pkl")

documents[0].metadata

documents[0].page_content

# CONCLUSIONS
# - Successfully scraped and processed web pages
# - Extracted relevant content for further vectorization and similarity search
