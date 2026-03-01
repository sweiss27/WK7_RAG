# Week 7: Retrieval-Augmented Generation (RAG) System (FAISS + GPT-2)

This project implements a simple Retrieval-Augmented Generation (RAG) prototype for an intelligent assistant.  
The system retrieves relevant text chunks from a small knowledge base using **FAISS** and **TF-IDF embeddings**, then uses **GPT-2** to generate a response grounded in the retrieved content. It also includes a basic qualitative evaluation.

---

## Project Overview

**Goal:**  
Build a RAG pipeline that:
1. Loads a small dataset to act as a knowledge base
2. Preprocesses and chunks text for retrieval
3. Builds a FAISS index over embeddings
4. Retrieves top-k relevant chunks for a query
5. Generates a response using GPT-2 based on retrieved chunks
6. Evaluates outputs using relevance, completeness, and coherence

**Notebook:**  
- `Week7_Retrieval-Augmented_Generation.ipynb`

---

## Approach (High Level)

### 1) Data Preprocessing
- Clean and normalize text (lowercasing, removing unwanted characters)
- Split text into chunks optimized for retrieval (chunk size can be adjusted)

### 2) Retrieval (FAISS + TF-IDF)
- Convert text chunks into TF-IDF vectors
- Convert vectors to dense float32 for FAISS
- Index vectors using FAISS for fast similarity search
- Retrieve top-k chunks per query

### 3) Generation (GPT-2)
- Concatenate retrieved chunks into a single prompt
- Use GPT-2 to generate an answer based on the retrieved context
- Tune generation parameters (max tokens, temperature, no-repeat ngrams)

### 4) Evaluation (Basic / Qualitative + Metrics)
Outputs are reviewed using:
- **Relevance** (cosine similarity)
- **Completeness** (keyword coverage)
- **Coherence** (perplexity)

---

## How to Run

### Option A: Run locally
1. Clone the repository
2. Open the notebook in Jupyter / VS Code
3. Run cells top-to-bottom

### Option B: Run in Google Colab
1. Upload `Week7_Retrieval-Augmented_Generation.ipynb` to Colab
2. Run all cells

> The notebook installs required libraries using `pip` if needed.

---

## Requirements

Typical libraries used:
- `datasets`
- `numpy`
- `scikit-learn`
- `faiss-cpu` (or `faiss-gpu` if available)
- `transformers`
- `torch`

Example install cell:
```bash
pip install datasets numpy scikit-learn faiss-cpu transformers torch
