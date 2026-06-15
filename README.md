# Career Intelligence RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that analyzes resumes against job descriptions using semantic search, vector embeddings, and Large Language Models (LLMs).

The platform combines LangChain, ChromaDB, Hugging Face embeddings, and Gemini 2.5 Flash to provide skill-gap analysis, resume improvement recommendations, and interview preparation insights.

## Features

- Resume-to-Job Description Matching
- Retrieval-Augmented Generation (RAG)
- Semantic Search with Vector Embeddings
- ChromaDB Vector Database
- Hugging Face Sentence Transformers
- LangChain-Based Document Processing
- Skill Gap Analysis
- Resume Improvement Suggestions
- Interview Preparation Recommendations
- Interactive Streamlit Interface

## Tech Stack

### Frontend

- Streamlit

### Generative AI

- Gemini 2.5 Flash
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)

### Vector Search & Embeddings

- ChromaDB
- Hugging Face Sentence Transformers
- Semantic Search
- Vector Embeddings

### Frameworks & Libraries

- LangChain
- PyPDF

### Language

- Python

## Architecture

Resume PDF + Job Description PDF → Document Processing → Text Chunking → Embedding Generation → ChromaDB Vector Database → Semantic Retrieval → Context Augmentation → Gemini 2.5 Flash → Analysis & Recommendations

## How It Works

1. Users upload a Resume PDF and a Job Description PDF.
2. The application extracts and processes document content.
3. Documents are split into smaller chunks using LangChain.
4. Hugging Face sentence-transformer models generate vector embeddings.
5. Embeddings are stored inside ChromaDB.
6. User queries are converted into embeddings and matched using semantic similarity search.
7. Relevant document chunks are retrieved and injected into the prompt.
8. Gemini 2.5 Flash generates grounded responses using the retrieved context.

## Example Questions

- What skills am I missing for this role?
- How well does my resume match the job description?
- What interview topics should I prepare?
- Which qualifications are required but missing?
- How can I improve my resume?

## Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Embeddings
- Semantic Search
- Prompt Engineering
- Context Augmentation
- LangChain Workflows
- Hugging Face Models
- Gemini API Integration

## Future Enhancements

- Multi-document knowledge bases
- Persistent vector storage
- Conversational memory
- Hybrid Search (Keyword + Semantic)
- Resume Scoring System
- AI Mock Interview Assistant

## Author

**Pranshu Malhotra**

B.Tech Computer Science and Engineering
