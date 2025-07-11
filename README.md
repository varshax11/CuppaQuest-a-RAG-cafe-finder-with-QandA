# ☕ CuppaQuest - Find the Cutest Cafes in Chennai!

Welcome to **CuppaQuest**, a charming Streamlit web app that helps you discover the cutest cafes in Chennai using a combination of PDF-based cafe data, RAG (Retrieval-Augmented Generation), and a hybrid retriever powered by HuggingFace and LangChain

<img width="1292" alt="Screenshot 2025-07-06 at 10 49 14 AM" src="https://github.com/user-attachments/assets/2f90ad85-003c-4750-b92d-8b410120284f" />
<img width="1297" alt="Screenshot 2025-07-06 at 10 41 38 AM" src="https://github.com/user-attachments/assets/e72c86aa-7172-4288-94ca-8388e582e4e6" />

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













