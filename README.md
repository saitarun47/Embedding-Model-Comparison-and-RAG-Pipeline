# Embedding-Model-Comparison-and-RAG-Pipeline

NAME : SK Sai Tarun
Email : tanuj00047@gmail.com


# Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to compare the performance of two embedding models—Cohere AI and Voyage AI—in the context of financial question-answering.


# Installation

Clone the repository

Create and activate a virtual environment

Install the required dependencies: pip install -r requirements.txt

Set up environment variables:Create a .env file in the project root directory and add your API keys:

COHERE_API_KEY=your_cohere_api_key

VOYAGE_API_KEY=your_voyage_api_key

PINECONE_API_KEY=your_pinecone_api_key

GROQ_API_KEY=your_groq_api_key

ATHINA_API_KEY=your_athina_api_key


# Usage

1.Run cohere_emb.py  : python cohere_emb.py

2.Run voyage.py      : python voyage.py

3.Run rag.py         : python rag.py


# Project Structure

voyage.py & cohere_emb.py : Prepares the FinanceBench dataset, generates embeddings using Voyage AIand cohere, and indexes them in Pinecone.

rag.py: Implements the RAG pipeline, performing document retrieval and answer generation using both Cohere AI and Voyage AI embeddings.

evaluation.py: Contains functions to compute evaluation metrics (EM, F1 Score, BLEU, ROUGE) for the generated answers.

requirements.txt: Lists all the Python dependencies required for the project.
