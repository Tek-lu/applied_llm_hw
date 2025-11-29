# Code Plagiarism Detection System

Course: Applied LLM Systems  
Assignment: Homework 1 - Code Plagiarism Detection with RAG Systems

## Overview

This project implements four different approaches to detect code plagiarism: pure embedding search, direct LLM analysis, standard RAG, and hybrid RAG. The goal is to compare their effectiveness and understand the trade-offs between retrieval quality, computational cost, and accuracy.

## Data Collection

Source Repository

I chose the [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) repository as my data source for several reasons:

1. Algorithm-focused: The repository contains implementations of fundamental algorithms. Since algorithms can be implemented in multiple ways while maintaining the same core logic, they're perfect for testing plagiarism detection.

2. Short files: Most implementations are 20-100 lines, which makes them easy to send to LLMs without hitting token limits.

3. Educational context: These are the types of algorithms students commonly implement in courses, making this a realistic plagiarism detection scenario.

4. Self-contained: Each file typically contains one complete algorithm, which works well with file-level chunking.

### Data Extraction Process

1. Downloaded the repository as a ZIP file from GitHub
2. Extracted the archive
3. Selected 5 projects/folders that I wanted to use:
   - `web_programming/`
   - `strings/`
   - `hashes/`
   - `bit_manipulation/`
   - `backtracking/`
4. Deleted all non-Python files from these folders (kept only `.py` files)
5. Moved the cleaned folders into `data/reference_corpus/`

Repository Links:
- Main: https://github.com/TheAlgorithms/Python/tree/master
- Web Programming: https://github.com/TheAlgorithms/Python/tree/master/web_programming
- Strings: https://github.com/TheAlgorithms/Python/tree/master/strings
- Hashes: https://github.com/TheAlgorithms/Python/tree/master/hashes
- Graphs: https://github.com/TheAlgorithms/Python/tree/master/bit_manipulation
- Backtracking: https://github.com/TheAlgorithms/Python/tree/master/backtracking

### Chunking Decision: File-Level

I decided to use file-level chunking instead of function-level chunking because:

- After cleaning (removing comments and whitespace), most files are small enough to fit in LLM context windows
- Each file usually contains one complete algorithm implementation, so it's a natural semantic unit
- File-level chunking preserves the full context including helper functions and imports
- It's simpler to implement and avoids breaking related code across chunks

### Test Dataset

I created a test dataset with 30+ labeled cases in `data/test_dataset.json`:

Positive Examples (Plagiarized):
- Took functions from the reference corpus
- Applied realistic transformations:
  - Variable renaming (e.g., `count` → `total`)
  - Comment removal
  - Minor refactoring (e.g., loops to list comprehensions)
  - Structure reordering

Negative Examples (Original):
- Wrote original code for similar tasks with different logic
- Used code from unrelated problems

Each case includes:
```json
{
  "id": "case_001",
  "query_code": "actual code to test",
  "label": "plagiarized" or "original",
  "transformation_type": "variable_renaming",
  "source_file": "path/to/source.py"
}
```

## Project Structure

```
APPLIED_AI_PROJECT/
├── data/
│   ├── reference_corpus/          # Downloaded algorithm code
│   │   ├── web_programming/
│   │   ├── strings/
│   │   ├── hashes/
│   │   ├── graphs/
│   │   └── backtracking/
│   └── test_dataset.json          # My labeled test cases
├── indexes/
│   ├── embeddings.npy             # Pre-computed embeddings
│   └── clean_texts.pkl            # Cleaned code
├── results/
│   ├── metrics_summary.json
│   ├── metrics_summary.csv
│   ├── method_comparison.png      # Required comparison chart
│   ├── rag_ablation.json
│   ├── hybrid_ablation.json
│   ├── failure_analysis.json
│   └── final_conclusion.txt
├── 01_indexing.ipynb              # Build indexes
├── 02_interactive.ipynb           # Core functions
├── 03_evaluation.ipynb            # Evaluate and analyze
├── plagiarism_detectors.py        # I had to extract code with py extention becouse otherwise it was not visible by 03_evaluation.ipynb
├── requirements.txt
├── .env                           # API key (not in git)
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone and navigate to the project:
```bash
git clone <repo-url>
cd APPLIED_AI_PROJECT
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API key:
```
OPENAI_API_KEY=your_key_here
```

## Usage

### Step 1: Build Indexes (01_indexing.ipynb)

This notebook:
- Loads all `.py` files from `data/reference_corpus/`
- Cleans code by removing comments and extra whitespace
- Generates embeddings using `all-MiniLM-L6-v2`
- Builds BM25 index for hybrid retrieval
- Saves everything to `indexes/` folder

Just run all cells from top to bottom.

### Step 2: Interactive Testing (02_interactive.ipynb)

This notebook lets you test individual code snippets with all four methods:

```python
from plagiarism_detectors import (
    detect_embedding,
    detect_llm,
    detect_rag,
    detect_hybrid_rag
)

test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Test with different methods
result = detect_embedding(test_code, top_k=5, threshold=0.75)
result = detect_llm(test_code, max_files=5)
result = detect_rag(test_code, top_k=5)
result = detect_hybrid_rag(test_code, top_k_embeddings=5, top_k_bm25=5)
```

### Step 3: Evaluation (03_evaluation.ipynb)

This notebook:
- Loads the test dataset
- Runs all four methods on every test case
- Calculates precision, recall, F1, and accuracy
- Performs ablation studies (varying k and weights)
- Generates visualizations
- Saves all results to `results/` folder

The main output is `results/method_comparison.png` which shows the performance comparison.

## Detection Methods

### 1. Pure Embedding Detection

How it works:
1. Converts test code to an embedding vector
2. Computes cosine similarity with all reference embeddings
3. Returns top-k most similar files
4. Flags as plagiarized if similarity > threshold

Pros: Very fast, no API costs  
Cons: Misses heavily refactored code  
Best for: Quick screening

### 2. Direct LLM Analysis

How it works:
1. Sends test code + some reference files directly to LLM
2. Asks the model to determine if it's plagiarized
3. Returns the LLM's reasoning

Pros: Deep understanding, catches refactoring  
Cons: Slow, expensive, limited by context size  
Best for: Detailed analysis of specific cases

### 3. Standard RAG

How it works:
1. Uses embeddings to retrieve top-k similar files
2. Sends only those retrieved files + test code to LLM
3. LLM makes the final decision

Pros: Good balance of speed, cost, and accuracy  
Cons: Quality depends on retrieval  
Best for: General-purpose detection

### 4. Hybrid RAG

How it works:
1. Retrieves using both embeddings (semantic) and BM25 (lexical)
2. Combines both scores with weighted fusion
3. Sends fused results to LLM for final decision

Pros: Best accuracy, catches different plagiarism types  
Cons: More complex, slightly slower  
Best for: When accuracy is most important

## Implementation Details

### Code Cleaning
```python
def clean_code(code):
    # Remove multi-line comments
    code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    return code
```

### Embedding Model
- Using `all-MiniLM-L6-v2` from sentence-transformers
- Produces 384-dimensional vectors
- Fast enough for real-time use

### LLM Setup
- Primary model: `gpt-4o-mini` (good balance of cost and quality)
- Alternative: `gpt-3.5-turbo` (cheaper)
- API key loaded from environment variables

## Evaluation Approach

### Metrics
- Precision: Of flagged cases, how many were actually plagiarized?
- Recall: Of actual plagiarism, how much did we catch?
- F1 Score: Balance between precision and recall
- Accuracy: Overall correctness

### Ablation Studies

RAG - Varying k:
- Tested k = [1, 3, 4, 5]
- Shows how number of retrieved documents affects performance

Hybrid RAG - Varying weights:
- Tested embedding weight α = [0.2, 0.4, 0.6, 0.8]
- BM25 weight = 1 - α
- Finds optimal balance between semantic and lexical matching

### Failure Analysis

The evaluation notebook analyzes:
- False Positives: Why did we incorrectly flag original code?
- False Negatives: Why did we miss actual plagiarism?
- Patterns by transformation type

Results are saved in `results/failure_analysis.json`.

## Trade-offs and Recommendations

### Key Trade-offs

| Method | Speed | Cost | Accuracy | Scalability |
|--------|-------|------|----------|-------------|
| Embedding | Fast | Low | Medium | Excellent |
| Direct LLM | Slow | High | High | Poor |
| Standard RAG | Medium | Medium | High | Good |
| Hybrid RAG | Medium | Medium | Highest | Good |

### When to Use Each Method

Embedding onl: 
- First-pass screening of many submissions
- When speed matters more than perfect accuracy
- Budget-constrained scenarios

Direct LLM: 
- Small number of files (<50)
- When you need detailed explanations
- Investigating specific suspicious cases

Standard RAG: 
- Default choice for most use cases
- Good balance of everything
- Medium to large code repositories

Hybrid RAG: 
- When maximum accuracy is needed
- High-stakes decisions (grading, legal issues)
- When you expect diverse plagiarism patterns

### My Recommendations

1. Start with embedding-only as a fast filter
2. Use standard RAG for routine checking
3. Use hybrid RAG for final decisions on flagged cases
4. Keep direct LLM for manual review of edge cases

Based on my ablation studies, I found that:
- RAG works best with k=3-5 documents
- Hybrid RAG performs best with α≈0.6 (60% embedding, 40% BM25)

