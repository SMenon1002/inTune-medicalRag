from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
    
    @abstractmethod
    def compute(self, 
                test_case: Dict[str, Any], 
                ground_truth: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """
        Compute the metric for a single test case.
        
        Args:
            test_case: The test case to evaluate
            ground_truth: The ground truth data for this test case
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            Dictionary containing the metric results
        """
        pass
    
    @abstractmethod
    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results across multiple test cases.
        
        Args:
            results: List of individual test case results
            
        Returns:
            Dictionary containing aggregated metrics
        """
        pass
    
    def run(self, 
            test_cases: List[Dict[str, Any]], 
            ground_truth: Dict[str, Any],
            **kwargs) -> Dict[str, Any]:
        """
        Run the metric evaluation on all test cases.
        
        Args:
            test_cases: List of test cases to evaluate
            ground_truth: Ground truth data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing both individual and aggregated results
        """
        individual_results = []
        
        for test_case in test_cases:
            result = self.compute(test_case, ground_truth, **kwargs)
            individual_results.append(result)
        
        aggregated_results = self.aggregate(individual_results)
        
        self.results = {
            "metric_name": self.name,
            "individual_results": individual_results,
            "aggregated_results": aggregated_results
        }
        
        return self.results 