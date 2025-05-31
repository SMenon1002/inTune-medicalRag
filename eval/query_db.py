import sys
import json
import time
from pathlib import Path
import os
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from rag_prep import RAGPreprocessor
# from config import GEMINI_API_KEY
import sys
sys.path.append("..")  # Add parent directory to path
from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from rag_prep import RAGPreprocessor
# from eval.config import EvalConfig, default_config

class VectorDBQuerier:
    def __init__(self):
        self.rag_prep = RAGPreprocessor()
        self.results_dir = Path("eval/manual_queries")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def query_db(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector DB and return results with metadata."""
        results = self.rag_prep.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results for better readability
        formatted_results = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            formatted_results["results"].append({
                "rank": i + 1,
                "text": doc,
                "metadata": metadata,
                "similarity_score": float(1 - distance)  # Convert distance to similarity
            })
            
        return formatted_results
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save query results to a timestamped JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"query_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        return filepath
    
    def print_results(self, results: Dict[str, Any]):
        """Print query results in a readable format."""
        print("\n" + "="*80)
        print(f"Query: {results['query']}")
        print(f"Time: {results['timestamp']}")
        print("="*80)
        
        for result in results["results"]:
            print(f"\nRank {result['rank']} (Similarity: {result['similarity_score']:.3f})")
            print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
            print("-"*80)
            print(result['text'])
            print("-"*80)

def main():
    querier = VectorDBQuerier()
    
    while True:
        print("\nEnter your query (or 'q' to quit):")
        query = input("> ").strip()
        
        if query.lower() == 'q':
            break
            
        try:
            print("\nNumber of results to return (default: 5):")
            n_input = input("> ").strip()
            n_results = int(n_input) if n_input else 5
            
            # Get and display results
            results = querier.query_db(query, n_results)
            querier.print_results(results)
            
            # Save results
            filepath = querier.save_results(results)
            print(f"\nResults saved to: {filepath}")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 