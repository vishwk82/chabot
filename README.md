# 🧠 Ubuntu Q&A Chatbot

A FastAPI-based Q&A chatbot that uses OpenAI embeddings, Chroma vector store, and a reranker (HuggingFace cross-encoder) to answer Ubuntu-related queries using markdown documents.

---

## 📦 Project Structure

```
.
├── app.py                     # FastAPI app with /chat endpoint
├──build_vector_store.py      # Script to process and embed markdown documents
├── config.py                 # Configuration for embeddings, LLM, paths, etc.
├── logger_config.py          # Logging setup
├── requirements.txt
├── Dockerfile
├── data/                     # Folder containing markdown files (*.md)
└── embeddings_store/         # Folder where vector store is persisted
```

---

## 🚀 Features

- Accepts Ubuntu-related questions and responds using vector-based retrieval.
- Uses **OpenAI Embeddings (`text-embedding-ada-002`)**.
- Stores vectors using **ChromaDB**.
- Implements **contextual compression retriever with reranker**.
- Supports **conversational memory** with LangChain.
- Provides a Swagger UI at `/docs` for interactive API testing.
- Containerized with Docker.

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/vishwk82/chabot.git
cd chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API Key

```bash
export OPENAI_API_KEY=your-api-key-here
```

You can also create a `.env` file or set it in the Docker container as shown below.

---

## 📚 Preparing the Vector Store

Place your `.md` files in the `./data/` directory, then run:

```bash
python build_vector_store.py
```

This script will:
- Load markdown files
- Split into chunks
- Generate embeddings
- Save vector store to `./embeddings_store/`

---

## ▶️ Running the API

```bash
uvicorn app:app --reload
```

Visit Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Sample Request

### Endpoint

```
POST /chat
```

### Request Body

```json
{
  "question": "How do I update packages in Ubuntu?"
}
```

### Response

```json
{
  "answer": "In Ubuntu Core, packages are managed and updated through snap packages. Updates are automatic, reliable, secure, and transparent, governed by snapd, the snap daemon. You can see which snap packages are installed by using the command `snap list`. Updates to snaps, including security updates, are published in the Ubuntu store and automatically applied to your system.",
  "sources": [
    {
      "source": "update_instructions.md",
      "content": "To update all packages in Ubuntu, run `sudo apt update && sudo apt upgrade`."
    }
  ]
}
```


## ✅ Requirements

- Python 3.9+
- OpenAI API key
- Dependencies listed in `requirements.txt`

---

## 📄 License

MIT License