import sys
import json
import time
from pathlib import Path
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from rag_prep import RAGPreprocessor
# from eval.config import EvalConfig, default_config


def extract_and_store_pdf_text():
    """Extract text from PDFs and store in a structured format."""
    # Initialize RAG preprocessor
    rag_prep = RAGPreprocessor()
    
    # Create output directory
    output_dir = Path("eval/pdf_extracts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Format results
        formatted_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": {}
        }
        
        # Get all documents from the collection
        results = rag_prep.collection.get()
        
        # Group by source document
        documents_by_source = {}
        for doc, metadata in zip(results['documents'], results['metadatas']):
            source = metadata['source']
            if source not in documents_by_source:
                documents_by_source[source] = {
                    'pages': {}
                }
            documents_by_source[source]['pages'][metadata['page']] = {
                'text': doc,
                'metadata': metadata
            }
        
        # Store in formatted results
        formatted_results['files'] = documents_by_source
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"pdf_extracts_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(formatted_results, f, indent=2)
            
        print(f"\nExtracted text saved to: {output_file}")
        print("\nSummary:")
        for source, content in documents_by_source.items():
            print(f"\n{source}:")
            print(f"- Pages: {len(content['pages'])}")
        
        return formatted_results
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None

if __name__ == "__main__":
    extract_and_store_pdf_text() 