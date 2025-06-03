# System Approach

The system implements a Retrieval-Augmented Generation (RAG) pipeline specifically designed for medical information retrieval and response generation. Here's the complete workflow:

## 1. Document Processing
- **PDF Ingestion**: Medical guidelines in PDF format are processed using multiple PDF libraries
  - Initially attempted to use PyPDF2 for basic text extraction
  - Added pdfplumber for better table extraction capabilities
  - Attempted to integrate image extraction using pdf2image and Pillow
  - Note: Image extraction and Gemini Vision integration was attempted but not completed due to API call failures and time constraints
  - Currently focuses on text and table extraction with source attribution

## 2. Text Chunking and Embedding
- **Semantic Chunking**: Implemented in `rag_prep.py`
  - Splits documents into semantically meaningful chunks
  - Preserves context and medical terminology
  - Handles medical abbreviations and special formatting
  - Maintains source and page information for each chunk

- **Embedding Generation**: Using Gemini's embedding model
  - Converts text chunks into vector embeddings
  - Uses Gemini's embedding model for semantic representation
  - Enables vector-based similarity search through ChromaDB

## 3. Vector Storage
- **ChromaDB Integration**:
  - Used because its open source and there were time constraints, didn't think too much about this one
  - Provides built-in semantic similarity search
  - Uses cosine similarity for vector comparison
  - Maintains metadata for source attribution
  - Provides persistent storage for long-term use

## 4. Query Processing
- **User Query Handling**:
  - Receives natural language queries about hypertension
  - Converts queries into embeddings
  - Performs similarity search against stored documents
  - Retrieves most relevant chunks with source information

## 5. Response Generation
- **Context Assembly**:
  - Combines retrieved chunks with source attribution
  - Formats context for the language model

- **LLM Response Generation**:
  - Uses Gemini AI for response generation

## 6. User Interface
- **Streamlit Web Application**:
  - Was quick to get off the ground
  - Provides intuitive chat interface

## 7. Evaluation and Quality Control
- **Response Evaluation**:
  - Ground truth came from chatgpt and was manually verified
  - Model was tested on these metrics:
    1. Content Completeness: Measures if key information points are present, regardless of order
    2. Clinical Accuracy: Verifies if medical statements are factually correct
    3. Context Relevance: Ensures answers address the specific question asked
    4. Answer Clarity: Evaluates if information is understandable
    5. Source Adherence: Confirms information matches source material

  - These metrics were chosen because:
    - Content Completeness ensures no critical medical information is omitted
    - Clinical Accuracy is crucial for medical applications where incorrect information could be harmful
    - Context Relevance ensures the system provides focused, relevant answers to specific medical queries
    - Answer Clarity is important for medical information to be understood by both healthcare providers and patients
    - Source Adherence maintains the reliability of information by staying true to medical guidelines

  - LLM did the grading because:
    - Provides consistent evaluation across multiple responses
    - Can understand medical context and terminology
    - Can compare responses against ground truth systematically
    - Reduces human bias in evaluation
    - More efficient for large-scale testing

  - Test cases cover both adult and pediatric hypertension guidelines, including:
    - Clinical management approaches
    - Diagnostic criteria
    - Treatment recommendations
    - Patient care considerations
    - Financial and practical implementation aspects

  - Assesses accuracy against medical guidelines provided

## Key Technical Decisions
1. **Chunking Strategy**: Semantic chunking over fixed-size chunks to preserve medical context
2. **Embedding Model**: Gemini's embedding model - its free, it gets the job done and its pretty good for medical texts
3. **Vector Database**: ChromaDB for efficient similarity search and metadata storage and its easy to get going
4. **LLM Integration**: Gemini AI for response generation with source attribution
5. **Source Tracking**: Page-level metadata preservation throughout the pipeline {but i didnt end up showing it at the end}

## Data Flow
```
PDF Documents → Text Extraction → Semantic Chunking → Embedding Generation → Vector Storage
     ↑                                                                           ↓
User Query → Query Embedding → Similarity Search → Context Assembly → Response Generation
```

## Future Improvements
1. **Image Processing**:
   - Debug and implement Gemini Vision integration for medical images
   - Add support for figure and diagram analysis
   - Improve table extraction and processing through

2. **PDF Processing**:
   - Explore alternative PDF libraries for better image extraction
   - Implement OCR for scanned documents
   - Add support for more complex document layouts

3. **Content Enhancement**:
   - Add support for cross-referencing between images and text
   - Implement better handling of medical abbreviations
   - Improve table structure preservation



#Examples:

## Successes
Question:
What did GPs identify as a major challenge in translating hypertension guidelines into clinical practice?
Answer: GPs identified that effectively and efficiently implementing vital information and translating it into routine clinical care remains a great challenge.
Ground truth: "Effectively and efficiently implementing this vital information and translating it into routine clinical care remains the great challenge."


Question: 
Q: What did GPs suggest to improve patient willingness to participate in hypertension treatment?
A: To improve patient willingness to participate in hypertension treatment, GPs suggested:

    Patient education
    Use of cardiovascular risk assessment tools
    Self-monitoring of BP
    Consistent BP measurement and management

GT:
To improve patient willingness to participate in treatment, the following strategies were suggested: patient education, use of cardiovascular risk assessment tools, self monitoring of BP, and consistent BP measurement and management.

## Incomplete answers:
Q:What were the financial concerns raised by GPs regarding home BP monitoring?
A: Based on the information provided, GPs raised concerns about funding for home and ambulatory BP monitors.
GT: Home BP monitoring incurred a cost to practices because machines lent out were often not returned. From a business perspective practices did not want to charge patients for the hire of machines due to increased administration and goods and services tax implications.


## Failed cases:
Q: What were the four main themes identified by GPs regarding hypertension management in the study?
A: I am unable to provide the four main themes identified by GPs regarding hypertension management in the study. The document mentions that the four main themes are listed in Table 2, but the table itself is not included in the provided context.
GT: Four main themes were identified: uncertainty about blood pressure measurement, achieving consensus in practice, accommodating patient differences and addressing systematic barriers.

