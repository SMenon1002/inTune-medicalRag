# Medical RAG Chat Assistant

A Retrieval-Augmented Generation (RAG) based medical chat assistant that provides accurate information about hypertension using clinical practice guidelines.

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings and constants
├── exceptions.py         # Custom exception classes
├── load_embeddings.py    # Utility for loading embeddings and testing retrieval functionality
├── pdf_extractor.py      # PDF processing and text extraction
├── rag_prep.py          # RAG preprocessing and embedding generation
├── requirements.txt      # Project dependencies
├── eval/                # Evaluation scripts and metrics
├── src/                 # Nothing yet
└── chroma_db/           # Vector database storage
```

## Key Features

- **Semantic Chunking**: Intelligent text splitting based on semantic boundaries
- **Vector Storage**: Efficient document retrieval using ChromaDB
- **Context-Aware Responses**: Uses RAG to provide relevant medical information

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- GEMINI_API_KEY
- Other configuration settings in config.py

3. Run the application:
```bash
streamlit run app.py
```

## Dependencies

- Streamlit
- Google Generative AI (Gemini)
- ChromaDB
- PyPDF2
- scikit-learn
- tiktoken
- Other dependencies listed in requirements.txt