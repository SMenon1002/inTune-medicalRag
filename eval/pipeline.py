import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import asdict
from google import genai
from google.genai import types

import sys
sys.path.append("..")  # Add parent directory to path
from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL
from rag_prep import RAGPreprocessor
from eval.config import EvalConfig, default_config

SYSTEM_PROMPT = """**System instructions:**
You are a helpful medical assistant. Your role is to provide accurate medical information based on authoritative sources.
You must:
- Only use information from the provided context
- Say when you cannot find relevant information
- Use clear, professional medical terminology while remaining accessible
- Never make up or infer information not present in the context
- Focus on accuracy and factual information"""

class EvalPipeline:
    def __init__(self, config: EvalConfig = default_config):
        self.config = config
        self.metrics = {}
        self.results = {
            "metadata": {
                "timestamp": time.time(),
                "config": self._get_json_safe_config()
            },
            "evaluations": {}
        }
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.rag_prep = RAGPreprocessor()
        
        # Create results directory if it doesn't exist
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.result_file = self.config.results_dir / f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
    def _get_json_safe_config(self) -> Dict[str, Any]:
        """Convert config to JSON-safe dictionary."""
        config_dict = asdict(self.config)
        config_dict['test_data_path'] = str(config_dict['test_data_path'])
        config_dict['ground_truth_path'] = str(config_dict['ground_truth_path'])
        config_dict['results_dir'] = str(config_dict['results_dir'])
        return config_dict
        
    def _save_current_results(self):
        """Save current evaluation results to disk."""
        with open(self.result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def _parse_scores(self, scores_text: str) -> Tuple[List[float], str]:
        """Parse scores from Gemini response, handling various formats."""
        # Clean up the text
        scores_text = scores_text.strip()
        
        # Split on newlines and clean each line
        score_lines = [line.strip() for line in scores_text.split('\n') if line.strip()]
        
        # Try to parse each line as a float
        scores = []
        for line in score_lines:
            try:
                # First try direct float conversion
                score = float(line)
                if 0 <= score <= 1:  # Validate score is in range
                    scores.append(score)
                    continue
            except ValueError:
                pass
                
            try:
                # If that fails, try cleaning the line first
                # Remove any leading numbers or dots that might be part of formatting
                # e.g., "1. 0.8" -> "0.8"
                cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                score = float(cleaned_line)
                if 0 <= score <= 1:  # Validate score is in range
                    scores.append(score)
            except ValueError:
                continue
                
        return scores, scores_text

    def load_test_data(self) -> Dict[str, Any]:
        """Load test cases from the configured test data path."""
        with open(self.config.test_data_path) as f:
            return json.load(f)
            
    def load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data for evaluation."""
        with open(self.config.ground_truth_path) as f:
            return json.load(f)
    
    def get_rag_response(self, query: str, context: str = None) -> Tuple[str, Dict[str, Any]]:
        """Generate a response using the RAG system."""
        # Get relevant context from ChromaDB if not provided
        retrieved_context = {}
        if context is None:
            results = self.rag_prep.collection.query(
                query_texts=[query],
                n_results=3
            )
            context = "\n".join(results['documents'][0])
            retrieved_context = {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0]
            }
            
        formatted_prompt = f"{SYSTEM_PROMPT}\n\n**Context:**\n{context}\n\n**User query:**\n{query}"
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=formatted_prompt)],
            )
        ]
        
        generate_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        
        response_stream = self.client.models.generate_content_stream(
            model=GEMINI_CHAT_MODEL,
            contents=contents,
            config=generate_config,
        )
        
        return "".join(chunk.text for chunk in response_stream if chunk.text), retrieved_context

    def evaluate_response(self, 
                         response: str, 
                         ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single response against ground truth using Gemini."""
        eval_prompt = f"""You are an expert evaluator for medical question-answering systems. Rate the following response on a scale from 0 to 1 (where 1 is perfect) for each criterion. Return only the scores as numbers, one per line.

Ground Truth Answer: {ground_truth['answer']}
System Response: {response}

Evaluation Criteria:
1. Content Completeness: Are all key information points from the ground truth present in the response?
Required elements: {ground_truth.get('required_elements', [])}

2. Clinical Accuracy: Is the medical/clinical information factually correct compared to the ground truth?
Evaluation guideline: {ground_truth['evaluation_criteria'].get('clinical_accuracy', '')}

3. Context Relevance: Does the answer address the specific medical context asked about?
Evaluation guideline: {ground_truth['evaluation_criteria'].get('context_relevance', '')}

4. Answer Clarity: Is the information clear and understandable?
Evaluation guideline: {ground_truth['evaluation_criteria'].get('answer_clarity', '')}

5. Source Adherence: Does the information match the source material (ground truth)?
Evaluation guideline: {ground_truth['evaluation_criteria'].get('source_adherence', '')}

Return exactly five numbers between 0 and 1, one per line, without any additional text. Example:
0.85
0.92
0.78
0.95
0.88"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=eval_prompt)],
            )
        ]
        
        generate_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        
        response_stream = self.client.models.generate_content_stream(
            model=GEMINI_CHAT_MODEL,
            contents=contents,
            config=generate_config,
        )
        
        scores_text = "".join(chunk.text for chunk in response_stream if chunk.text)
        scores, original_text = self._parse_scores(scores_text)
        
        if len(scores) != 5:
            print(f"Error parsing Gemini scores: Expected 5 scores, got {len(scores)}. Original response:\n{original_text}")
            # Try to parse the original text again as a simple newline-separated list
            try:
                simple_scores = [float(s.strip()) for s in original_text.strip().split('\n') if s.strip()]
                if len(simple_scores) == 5 and all(0 <= s <= 1 for s in simple_scores):
                    scores = simple_scores
                    print("Successfully parsed scores from original text.")
                else:
                    print("Using fallback scoring.")
                    return {
                        "scores": {
                            "content_completeness": 0.0,
                            "clinical_accuracy": 0.0,
                            "context_relevance": 0.0,
                            "answer_clarity": 0.0,
                            "source_adherence": 0.0
                        },
                        "original_evaluation": original_text,
                        "used_fallback": True
                    }
            except ValueError:
                print("Using fallback scoring.")
                return {
                    "scores": {
                        "content_completeness": 0.0,
                        "clinical_accuracy": 0.0,
                        "context_relevance": 0.0,
                        "answer_clarity": 0.0,
                        "source_adherence": 0.0
                    },
                    "original_evaluation": original_text,
                    "used_fallback": True
                }
            
        return {
            "scores": {
                "content_completeness": scores[0],
                "clinical_accuracy": scores[1],
                "context_relevance": scores[2],
                "answer_clarity": scores[3],
                "source_adherence": scores[4]
            },
            "original_evaluation": original_text,
            "used_fallback": False
        }
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        start_time = time.time()
        
        # Load test data and ground truth
        test_data = self.load_test_data()
        ground_truth = self.load_ground_truth()
        
        # Run evaluation for each document
        for doc_id, doc_data in test_data["documents"].items():
            self.results["evaluations"][doc_id] = {
                "name": doc_data["name"],
                "questions": {}
            }
            
            # Evaluate each question
            for q_id, question in doc_data["questions"].items():
                try:
                    # Generate response using RAG
                    response, retrieved_context = self.get_rag_response(question)
                    
                    # Get corresponding ground truth
                    ans_id = f"ans{q_id[1:]}"  # Convert q1 to ans1
                    gt_answer = ground_truth["documents"][doc_id]["answers"][ans_id]
                    
                    # Evaluate response
                    evaluation = self.evaluate_response(response, gt_answer)
                    
                    # Store results
                    self.results["evaluations"][doc_id]["questions"][q_id] = {
                        "question": question,
                        "response": response,
                        "retrieved_context": retrieved_context,
                        "evaluation": evaluation,
                        "timestamp": time.time()
                    }
                    
                    # Save after each question is evaluated
                    self._save_current_results()
                    
                except Exception as e:
                    error_msg = f"Error processing question {q_id}: {str(e)}"
                    print(error_msg)
                    # Store error but continue processing
                    self.results["evaluations"][doc_id]["questions"][q_id] = {
                        "question": question,
                        "error": error_msg,
                        "timestamp": time.time()
                    }
                    self._save_current_results()
        
        # Add overall execution time
        self.results["metadata"]["execution_time"] = time.time() - start_time
        self._save_current_results()
        
        return self.results

if __name__ == "__main__":
    # Create and run the evaluation pipeline
    pipeline = EvalPipeline()
    results = pipeline.run_evaluation()
    print("Evaluation completed. Results saved to:", pipeline.result_file) 