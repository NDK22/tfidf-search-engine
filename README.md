# tfidf-search-engine

## Overview

This repository contains the implementation of a toy "search engine" in Python. The goal is to read a corpus, produce TF-IDF vectors for documents, and return the document with the highest cosine similarity score for a given query.

## Instructions

1. Install the Jupyter notebook viewer using the following commands:
    ```bash
    pip install jupyter
    pip install notebook  # Use "sudo" if installing at the system level
    ```

2. Run the Jupyter notebook viewer with the following command:
    ```bash
    jupyter notebook P1.ipynb
    ```
   This will start a web service at http://localhost:8888/ and display the instructions in the '.ipynb' file.

## Dataset

The assignment uses a corpus of 15 Inaugural addresses of different US presidents. The provided `.zip` file includes 15 `.txt` files.

## Programming Language and Modules

1. Utilize NLTK, a natural language processing toolkit for Python.
2. Install NLTK and import it into your `.py` file.
    ```bash
    pip install nltk
    ```
    In the Python interpreter:
    ```python
    import nltk
    nltk.download()
    ```

## Tasks

Your code should accomplish the following tasks:

1. Read the 15 `.txt` files and convert the text to lowercase.
2. Tokenize the content of each file using a regular expression tokenizer.
3. Perform stopword removal using NLTK's stopword list.
4. Perform stemming using NLTK's Porter stemmer.
5. Compute the TF-IDF vector for each document.
6. Implement query-document similarity using the `ltc.lnc` weighting scheme.

## What it returns

- `getidf(token)`: Return the inverse document frequency of a token.
- `getweight(filename, token)`: Return the normalized TF-IDF weight of a token in the document.
- `query(qstring)`: Return a tuple in the form of (filename, score) representing the document with the highest similarity to the query.

