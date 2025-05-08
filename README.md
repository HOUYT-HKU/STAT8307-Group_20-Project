# STAT8307 Group 20 Project
Graph-Based Keyword Extraction from Scientific Paper Abstracts using Word Embeddings

## 📁 File Overview

| Filename                        | Description |
|--------------------------------|-------------|
| `main.ipynb`                   | The main notebook for running the full keyword extraction pipeline. |
| `KeywordExtractor.py`          | Defines the baseline keyword extraction logic. |
| `KeywordExtractor_algorithm.py`| **Modified version** of the keyword extraction algorithm with enhancements. |
| `nlp_utils.py`                 | Contains utility functions for text preprocessing, tokenization, and embedding. |
| `nlp_utils_algorithm.py`       | **Improved version** of `nlp_utils.py` with modified NLP utility functions. |
| `data_utils.py`                | Functions for loading and preprocessing datasets. |
| `cluster.ipynb`                | Jupyter notebook for clustering analysis of extracted keywords. |
| `app.py`                       | The UI functions of our program you can upload documents and get the keyword. |
| `lexicons.py`                  | Provides keyword lexicons or dictionaries used in some extraction strategies. |


## ✅ Notes

- Files with `_algorithm` suffix are improved versions of the original algorithms.
- This repository also includes a `datasets/` folder (not shown here) to hold input datasets like `inspec`, `arxiv`, and `CS` datasets.

## 🚀 How to Run

You can start by opening and running `main.ipynb`. Make sure to install required dependencies listed in the notebook before executing.

---
