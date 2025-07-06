# ☕ CuppaQuest - Find the Cutest Cafes in Chennai!

Welcome to **CuppaQuest**, a charming Streamlit web app that helps you discover the cutest cafes in Chennai using a combination of PDF-based cafe data, RAG (Retrieval-Augmented Generation), and a hybrid retriever powered by HuggingFace and LangChain

<img width="1301" height="608" alt="Image" src="https://github.com/user-attachments/assets/d5ef2fd9-20ba-491d-ada5-1257af591341"

<img width="1297" height="589" alt="Image" src="https://github.com/user-attachments/assets/677c4ed5-62af-495e-957a-a83666f7d9fa"

---

## Features

- Explore curated cafe recommendations by location across Chennai
- Ask questions about cafes using natural language
- Powered by a hybrid BM25 + embedding retriever
- Works on custom documents (extracted from a PDF)
- Beautiful pink-themed UI with soft cafe vibes

---

## Tech Stack

- **Streamlit** for the web UI
- **LangChain** for RAG pipeline & prompt templating
- **HuggingFace** (Mistral 7B via Inference API) for the LLM
- **BM25 + FAISS** ensemble retriever
- **PDFLoader** for data ingestion
- **dotenv** for managing secrets

---

## File Structure

```bash
.
├── main.py                # Streamlit frontend
├── langchain_helper.py   # Backend RAG logic
├── cafeschennai.pdf      # Data source with cafe listings (must be in the correct path)
├── .env                  # HuggingFace token (HF_TOKEN)
└── README.md             # You're here!

```

---

## Getting started
### 1. Clone the repository

```bash

git clone https://github.com/your-username/cuppaquest.git
cd cuppaquest

```

### 2. Install dependencies

```bash

pip install -r requirements.txt

```

### 3. Setup HuggingFace Token

Create a .env file in the root folder:

```bash

HF_TOKEN=your_huggingface_token_here

```

---

## Usage

```bash

streamlit run main.py

```

Select a location from the dropdown to get the cafes for the location

Type a question (e.g., "What are the coffees popular in Nolita?") to get a contextual answer

---

## Liscense

This project is under MIT Liscense













