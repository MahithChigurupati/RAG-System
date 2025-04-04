# Simple RAG (Retrieval-Augmented Generation) System

A straightforward RAG system that processes PDF documents and answers questions about their content using OpenAI's GPT-4 and Cohere's reranking capabilities.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Cohere API key

## Setup

1. Clone the repository:
```bash
git clone git@github.com:MahithChigurupati/RAG-System.git
cd RAG-System
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # For Mac/Linux
# or
.venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example env file and update with your API keys:
```bash
cp .env.example .env
```

Example `.env` file:
```plaintext
OPENAI_API_KEY=sk-xxx...  # Your OpenAI API key
COHERE_API_KEY=xxx...     # Your Cohere API key
CHUNK_SIZE=1000          # Optional: Customize chunk size
CHUNK_OVERLAP=200        # Optional: Customize chunk overlap
```

5. Create a `content` directory for your PDFs:
```bash
mkdir content
```

## Usage

1. Place your PDF documents in the `content` directory
2. Run the system:
```bash
python rag_system.py
```

## Features

- PDF text extraction and processing
- Text chunking with configurable overlap
- Document embedding (OpenAI ada-002)
- Vector similarity search (FAISS)
- Result reranking (Cohere rerank-english-v3.0)
- Question answering (GPT-4)

## Example Queries

The system comes with default queries about:
- Personal information
- Professional achievements
- Certifications and qualifications
- Skills and experience
- Domain expertise (AI/ML, Blockchain)

You can modify queries in `rag_system.py`.

## Customization

Adjust these parameters in your `.env` file:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_TOKENS`: Maximum tokens for responses (default: 500)
- `TEMPERATURE`: Temperature for response generation (default: 0.7)

## Credits

Built with:
- LangChain
- OpenAI
- Cohere
- FAISS
- PyMuPDF