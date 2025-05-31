import os
import json
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.config import Settings
import tiktoken
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pdf_extractor import process_pdfs
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_TASK_TYPE,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    BATCH_SIZE,
    logger
)
from exceptions import EmbeddingError, ChromaDBError
load_dotenv()  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self._total_calls = 0
    
    def __call__(self, input: Documents) -> List[List[float]]:
        embeddings = []
        for text in tqdm(input, desc="Generating embeddings"):
            try:
                embedding = genai.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    content=text,
                    task_type=EMBEDDING_TASK_TYPE
                )
                embeddings.append(embedding["embedding"])
                self._total_calls += 1
            except Exception as e:
                error_msg = f"Error generating embedding: {str(e)}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg)
        
        logger.info(f"Generated {len(embeddings)} embeddings. Total calls: {self._total_calls}")
        return embeddings

class RAGPreprocessor:
    def __init__(self, 
                 max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
                 min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """Initialize the RAG preprocessor."""
        logger.info("Initializing RAGPreprocessor")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_text_processed = 0
        self.total_text_chunks = 0
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        try:
            # Initialize ChromaDB with persistent storage
            logger.info("Initializing ChromaDB")
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # Create embedding function
            self.embedding_function = GeminiEmbeddingFunction()
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Found existing collection '{COLLECTION_NAME}'")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created new collection '{COLLECTION_NAME}'")
        
        except Exception as e:
            error_msg = f"Error initializing ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise ChromaDBError(error_msg)

    def save_database(self):
        """Explicitly persist the database to disk."""
        self.chroma_client.persist()
        logger.info("Database persisted to disk")

    def export_to_json(self, output_file: str = "embeddings_backup.json"):
        """Export the collection data to a JSON file."""
        # Get all data from the collection
        collection_data = self.collection.get()
        
        # Prepare data for JSON serialization
        export_data = {
            "documents": collection_data["documents"],
            "metadatas": collection_data["metadatas"],
            "embeddings": collection_data["embeddings"],
            "ids": collection_data["ids"]
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(export_data, f)
        
        logger.info(f"Collection exported to {output_file}")
        return export_data

    @classmethod
    def load_from_json(cls, json_file: str, api_key: str):
        """
        Create a new RAGPreprocessor instance and load data from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing the exported data
            api_key: Gemini API key
        
        Returns:
            RAGPreprocessor instance with loaded data
        """
        # Create new instance
        instance = cls()
        
        # Load data from JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Add data to collection
        instance.collection.add(
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"],
            ids=data["ids"]
        )
        
        # Persist the loaded data
        instance.save_database()
        logger.info(f"Data loaded from {json_file} and persisted to database")
        return instance

    def get_token_count(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a piece of text."""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def calculate_similarity(self, embed1: List[float], embed2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return cosine_similarity([embed1], [embed2])[0][0]

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {type(text)}")
            return []

        # Handle common medical and academic abbreviations
        text = re.sub(r'(?<=Mr)\.(?=\s[A-Z])', '@', text)
        text = re.sub(r'(?<=Dr)\.(?=\s[A-Z])', '@', text)
        text = re.sub(r'(?<=Prof)\.(?=\s[A-Z])', '@', text)
        text = re.sub(r'(?<=et al)\.(?=\s[A-Z])', '@', text)
        text = re.sub(r'(?<=i\.e)\.(?=\s)', '@', text)
        text = re.sub(r'(?<=e\.g)\.(?=\s)', '@', text)
        text = re.sub(r'(?<=Fig)\.(?=\s\d)', '@', text)
        text = re.sub(r'(?<=Tab)\.(?=\s\d)', '@', text)
        text = re.sub(r'(?<=vs)\.(?=\s)', '@', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore periods and clean sentences
        sentences = [s.replace('@', '.').strip() for s in sentences if s.strip()]
        logger.debug(f"Split text into {len(sentences)} sentences")
        return sentences

    def find_semantic_boundary(self, sentences: List[str], start_idx: int) -> Tuple[int, float]:
        """
        Find the optimal semantic boundary in a window of sentences.
        Returns the index where the break should occur and the similarity score.
        """
        if len(sentences) - start_idx <= 1:
            return len(sentences), 1.0

        # Get embedding for the current sentence
        current_embed = self.get_embedding(sentences[start_idx])
        
        # Look ahead for semantic boundaries
        best_break = start_idx + 1
        best_score = float('inf')
        current_tokens = self.get_token_count(sentences[start_idx])
        
        for i in range(start_idx + 1, len(sentences)):
            # Check if adding next sentence exceeds max token limit
            next_tokens = self.get_token_count(sentences[i])
            if current_tokens + next_tokens > self.max_chunk_size:
                break
                
            # Get embedding for the next sentence
            next_embed = self.get_embedding(sentences[i])
            similarity = self.calculate_similarity(current_embed, next_embed)
            
            # Lower similarity indicates a potential boundary
            if similarity < best_score:
                best_score = similarity
                best_break = i
            
            current_tokens += next_tokens
            
        return best_break, best_score

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks using embedding-based similarity.
        """
        if not text:
            logger.warning("Empty text provided to chunk_text")
            return []

        original_length = len(text)
        logger.info(f"Starting text chunking for text of length {original_length} characters")
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_size = 0
        i = 0
        total_chars_processed = 0
        
        with tqdm(total=len(sentences), desc="Chunking text") as pbar:
            while i < len(sentences):
                if not current_chunk:
                    current_chunk.append(sentences[i])
                    current_size = self.get_token_count(sentences[i])
                    total_chars_processed += len(sentences[i])
                    i += 1
                    pbar.update(1)
                    logger.debug(f"Coverage: {(total_chars_processed/original_length)*100:.2f}% of text processed")
                    continue

                next_break, similarity = self.find_semantic_boundary(sentences, i)
                chunk_text = ' '.join(current_chunk)
                next_text = ' '.join(sentences[i:next_break])
                combined_size = self.get_token_count(chunk_text + ' ' + next_text)

                if similarity < self.similarity_threshold or combined_size > self.max_chunk_size:
                    if current_size >= self.min_chunk_size:
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_size = 0
                    else:
                        if i < len(sentences):
                            current_chunk.append(sentences[i])
                            total_chars_processed += len(sentences[i])
                            i += 1
                            pbar.update(1)
                else:
                    for j in range(i, next_break):
                        total_chars_processed += len(sentences[j])
                    current_chunk.extend(sentences[i:next_break])
                    current_size = self.get_token_count(' '.join(current_chunk))
                    i = next_break
                    pbar.update(next_break - i)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Calculate coverage metrics
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        coverage_percentage = (total_chunk_length / original_length) * 100
        average_chunk_size = total_chunk_length / len(chunks) if chunks else 0
        
        logger.info(f"Text chunking metrics:")
        logger.info(f"- Original text length: {original_length} characters")
        logger.info(f"- Total chunk length: {total_chunk_length} characters")
        logger.info(f"- Coverage: {coverage_percentage:.2f}%")
        logger.info(f"- Number of chunks: {len(chunks)}")
        logger.info(f"- Average chunk size: {average_chunk_size:.2f} characters")
        
        self.total_text_processed += original_length
        self.total_text_chunks += len(chunks)
        
        return chunks

    def process_pdf_content(self, pdf_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process PDF content and prepare documents for embedding."""
        documents = []
        total_text_length = 0
        processed_text_length = 0
        
        # First calculate total text length
        for pdf_name, content in pdf_results.items():
            if 'error' in content:
                continue
            for page_num, text in content['text'].items():
                total_text_length += len(text)
        
        logger.info(f"Total text to process: {total_text_length} characters")
        
        for pdf_name, content in pdf_results.items():
            if 'error' in content:
                logger.warning(f"Skipping {pdf_name} due to error: {content['error']}")
                continue
            
            # Process text content
            for page_num, text in content['text'].items():
                chunks = self.chunk_text(text)
                processed_text_length += len(text)
                
                coverage_percentage = (processed_text_length / total_text_length) * 100
                logger.info(f"Overall progress: {coverage_percentage:.2f}% of total text processed")
                
                for i, chunk in enumerate(chunks):
                    doc = {
                        'text': chunk,
                        'metadata': {
                            'source': pdf_name,
                            'page': page_num,
                            'chunk': i,
                            'type': 'text',
                            'length': len(chunk)
                        }
                    }
                    documents.append(doc)
            
            # Process tables
            for page_num, tables in content['tables']:
                for i, table in enumerate(tables):
                    table_str = table.to_string()
                    doc = {
                        'text': table_str,
                        'metadata': {
                            'source': pdf_name,
                            'page': page_num,
                            'table_num': i,
                            'type': 'table',
                            'length': len(table_str)
                        }
                    }
                    documents.append(doc)
        
        # Final coverage statistics
        total_processed_length = sum(len(doc['text']) for doc in documents)
        final_coverage = (total_processed_length / total_text_length) * 100 if total_text_length > 0 else 0
        
        logger.info("\nFinal Processing Statistics:")
        logger.info(f"- Total text length: {total_text_length} characters")
        logger.info(f"- Total processed length: {total_processed_length} characters")
        logger.info(f"- Final coverage: {final_coverage:.2f}%")
        logger.info(f"- Total documents created: {len(documents)}")
        logger.info(f"- Average document length: {total_processed_length/len(documents):.2f} characters")
        
        return documents

    def create_embeddings_and_store(self, documents: List[Dict[str, Any]]):
        """Create embeddings using Gemini and store in ChromaDB."""
        if not documents:
            logger.warning("No documents provided for embedding")
            return

        logger.info(f"Processing {len(documents)} documents")
        try:
            texts = [doc['text'] for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Add to ChromaDB in batches
            batch_size = 50
            for i in tqdm(range(0, len(texts), batch_size), desc="Storing in ChromaDB"):
                batch_end = min(i + batch_size, len(texts))
                self.collection.add(
                    documents=texts[i:batch_end],
                    ids=ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

            logger.info(f"Successfully processed and stored {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise

def main():
    try:
        logger.info("Starting PDF processing")
        pdf_files = [
            "Clinical Practice Guidelines _ Hypertension in children and adolescents.pdf",
            "hypertension_adults.pdf"
        ]

        # Verify files exist
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                raise FileNotFoundError(f"PDF file not found: {pdf_file}")

        pdf_results = process_pdfs(pdf_files)
        
        # Initialize RAG preprocessor
        rag_prep = RAGPreprocessor()
        
        # Process documents and create embeddings
        documents = rag_prep.process_pdf_content(pdf_results)
        logger.info(f"Created {len(documents)} document chunks")
        
        rag_prep.create_embeddings_and_store(documents)
        logger.info("Completed embedding creation and storage")
        
        # Test query
        logger.info("Testing embeddings with sample query")
        query = "What are the treatment guidelines for hypertension in children?"
        results = rag_prep.collection.query(
            query_texts=[query],
            n_results=3
        )
        
        print("\nExample query results:")
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\nResult {i+1}:")
            print(f"Source: {metadata['source']}, Page: {metadata['page']}")
            print(f"Text: {doc[:200]}...")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 