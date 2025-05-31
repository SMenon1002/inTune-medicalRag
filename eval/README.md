# RAG Evaluation Pipeline

This directory contains the evaluation pipeline for the medical RAG system. The pipeline assesses various aspects of the RAG system's performance.

## Structure

- `metrics/`: Contains individual metric implementations
- `data/`: Test datasets and ground truth
- `pipeline.py`: Main evaluation pipeline orchestrator
- `config.py`: Evaluation configuration
- `results/`: Directory for storing evaluation results

## Metrics

The evaluation pipeline includes:
1. Answer Relevance
2. Context Relevance
3. Retrieval Precision
4. Answer Correctness
5. Response Time

## Usage

Run the evaluation pipeline:
```bash
python eval/pipeline.py
``` 