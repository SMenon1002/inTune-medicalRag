from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import sys
sys.path.append("..")  # Add parent directory to path
from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL

@dataclass
class EvalConfig:
    # Data settings
    test_data_path: Path = Path("eval/data/test_cases.json")
    ground_truth_path: Path = Path("eval/data/ground_truth.json")
    results_dir: Path = Path("eval/results")
    
    # Metric settings
    metrics_to_run: List[str] = field(default_factory=lambda: [
        "content_completeness",
        "clinical_accuracy",
        "context_relevance",
        "answer_clarity",
        "source_adherence"
    ])
    
    # Evaluation parameters
    num_test_cases: int = 100
    similarity_threshold: float = 0.7
    max_response_time: float = 5.0  # seconds
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __post_init__(self):
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

# Default configuration
default_config = EvalConfig()

# API Keys
GEMINI_API_KEY = GEMINI_API_KEY
GEMINI_CHAT_MODEL = GEMINI_CHAT_MODEL 