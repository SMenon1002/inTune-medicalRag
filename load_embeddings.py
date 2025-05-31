import os
import logging
from dotenv import load_dotenv
from rag_prep import RAGPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embeddings(rag_prep: RAGPreprocessor):
    """Test the loaded embeddings with a sample query."""
    logger.info("Testing embeddings with sample query")
    query = "What are the recommended blood pressure thresholds for hypertension diagnosis?"
    results = rag_prep.collection.query(
        query_texts=[query],
        n_results=3
    )

    print(f"\nQuery: {query}")
    print("\nResults:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\nResult {i+1}:")
        print(f"Source: {metadata['source']}, Page: {metadata['page']}")
        print(f"Text: {doc[:200]}...")

def main():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")

        # Load embeddings from persistent storage
        logger.info("Loading embeddings from persistent storage")
        rag_prep = RAGPreprocessor(google_api_key=api_key)
        logger.info("Successfully loaded embeddings from persistent storage")

        # Test the loaded embeddings
        test_embeddings(rag_prep)

    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 