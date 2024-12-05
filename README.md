# RAG_document_QA_with_GROQ

This project creates a RAG (Retrieval-Augmented Generation) Q&A System utilizing Groq AI's Llama3-8b-8192 model. It integrates document ingestion, vector embeddings, and retrieval capabilities to answer user queries based on research papers. Key functionalities include:

1) Document Ingestion & Splitting: Loads research papers in PDF format, splits them into manageable chunks using RecursiveCharacterTextSplitter.
2) Vector Store Creation: Converts documents into vector embeddings using Ollama Embeddings and stores them in a FAISS vector database.
3) RAG Pipeline: Combines the LLM with a retrieval mechanism to ground responses in context from stored vectors.
4) Streamlit Interface: Provides a user-friendly interface for query input, document embedding initialization, and similarity search exploration.
5) This system is designed for retrieving contextually accurate answers while enabling the exploration of similar document content.







