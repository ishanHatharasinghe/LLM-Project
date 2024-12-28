# Import Libraries and Modules
import os
import pathlib
import textwrap
from typing import List

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from IPython.display import Markdown, display
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import (PyPDFDirectoryLoader,
                                                  PyPDFLoader, TextLoader)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, pipeline)


# Define the RAG function
def rag(llm, rag_chain, vector_db, return_source_documents, chain_type_kwargs):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=rag_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=return_source_documents,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa


# Load the environment variables from the .env file
load_dotenv()

# Set the Hugging Face token from the environment variable
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("BOT3_APIKEY")

# Initialize the LLM with a positive temperature
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.getenv("BOT3_APIKEY"),
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={
        "min_length": 50,
        "max_length": 500,  # Adjust this to keep total token count within limits
        "temperature": 0.1,  # Set to a small positive value
        "max_new_tokens": 1000,  # Reduced max_new_tokens
        "num_return_sequences": 1,
    },
)


# Use a raw string for the file path to avoid issues with backslashes
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")

# Load the PDF data
pdf_data = loader.load()

# Initialize the text splitter with the desired chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)

# Split the loaded PDF data into chunks
chunks = text_splitter.split_documents(pdf_data)

# Initialize embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="impira/layoutlm-document-qa")
chunk_embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

# Initialize the vector database
vector_db = FAISS.from_documents(chunks, embeddings_model)

# Create the retriever
retriever = vector_db.as_retriever()

# Define the system prompt
system_prompt = (
    "You are an intelligent chatbot that answers only using the information provided in the loaded data."
    " If the loaded data does not contain any relevant information for the question, respond first 'no idea,' and followed by a general knowledge answer."
    " Do not blend the loaded data with general knowledge when answering."
    "\n\n"
    "responses need to short for only 20 words"
    "Context: {context}"
    "\n"
    "Note: Provide a general knowledge answer if the loaded data does not contain the required information, but clearly separate the two responses."
)


# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

history = []


def qa(question):
    # Build the history string
    history_str = ""
    for h in history:
        history_str += f"q: {h['question']}, a: {h['answer']}\n"

    # Generate the prompt text
    prompt_text = prompt_template.format(input=question, context="\n\n".join([
                                         doc.page_content for doc in chunks]))

    # Generate the response from the model
    response = llm(prompt_text)

    # Extract the answer based on common response patterns
    if "no idea" in response.lower():
        # Get the response after "Chatbot: "
        final_answer = response.split("Human: ")[-1].strip()
    else:
        # Return the full response if "no idea" is not found
        final_answer = response.strip()

    # Update history with the new question and extracted answer
    history.append({"question": question, "answer": final_answer})

    return final_answer


def load_pdf_and_answer(question):
    # Call the qa() function
    return qa(question)


# Create the question-answering chain
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# Create the RAG chain
rag_chain = create_retrieval_chain(retriever, qa_chain)
