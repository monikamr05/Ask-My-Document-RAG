---
title: Ask My Documents - AI RAG Assistant
emoji: 📑
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 📑 Ask My Documents – AI RAG Assistant

A beginner-friendly, production-ready **Retrieval-Augmented Generation (RAG)** web application built with **Python**, **Gradio**, **LangChain**, **FAISS**, **Hugging Face sentence-transformers**, and the **OpenRouter API**.

Live Demo: https://ask-my-documents-rag-roeh.onrender.com

---

## 🌟 Key Features

1. **Multi-Format Document Upload**: Seamlessly ingest and parse **PDF**, **DOCX**, and **TXT** files.
2. **Smart Semantic Chunking**: Splits large documents into overlapping semantic chunks with rich metadata (filename, page numbers, chunk ID).
3. **Local Vector Embeddings**: Uses Hugging Face's lightweight `sentence-transformers/all-MiniLM-L6-v2` running locally on your CPU (fast, private, and free).
4. **Fast Similarity Retrieval**: In-memory **FAISS** (Facebook AI Similarity Search) index for sub-millisecond semantic search.
5. **OpenRouter LLM Generation**: Configurable access to top AI models (e.g., `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`, `meta-llama/llama-3.3-70b-instruct:free`, `anthropic/claude-3.5-haiku`).
6. **Strict Grounding & Anti-Hallucination**: The AI answers questions **strictly based on the retrieved document content**. If the answer is not present in the document, it explicitly tells you so.
7. **Transparent Source Citation & Context Inspector**: View exact document names, page numbers, similarity scores, and text excerpts used to formulate each answer.
8. **Modern Gradio Web UI**: Clean responsive interface with document statistics, model settings, interactive chat history, and one-click reset.

---

## 🏗️ How RAG Works (Architecture Overview)

```
[ User Uploads Documents (.pdf, .docx, .txt) ]
                      │
                      ▼
[ Document Parsing & Metadata Extraction (pypdf, python-docx) ]
                      │
                      ▼
[ Recursive Text Chunking (LangChain Text Splitter) ]
                      │
                      ▼
[ Embedding Generation (Hugging Face sentence-transformers) ]
                      │
                      ▼
[ Vector Store Indexing (FAISS) ]
                      │
   ═══════════════════╪═════════════════════════════════════
                      │ (When User Asks a Question)
                      ▼
[ Semantic Similarity Search (FAISS Top-K Chunks) ]
                      │
                      ▼
[ Grounded Prompt Assembly (Context + Strict Rules + User Query) ]
                      │
                      ▼
[ OpenRouter API LLM Generation (e.g., GPT-4o-mini, Gemini 2.0) ]
                      │
                      ▼
[ Formatted Answer + Source Citations + Context Inspector ]
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- An **OpenRouter API Key** (Get a key from [openrouter.ai/keys](https://openrouter.ai/keys))

---

### 2. Installation

Clone or navigate to the project directory:
```bash
cd Ask_My_Document
```

Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

1. Copy the sample environment file:
   ```bash
   # Windows (Command Prompt / PowerShell)
   copy .env.example .env

   # macOS / Linux
   cp .env.example .env
   ```

2. Open `.env` in any text editor and add your OpenRouter API key:
   ```ini
   OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
   OPENROUTER_MODEL=openai/gpt-4o-mini
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   ```

> 💡 **Tip:** You can also enter or override your OpenRouter API key directly in the Gradio web interface settings accordion at runtime!

---

### 4. Run the Application

Launch the Gradio web server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## 📖 Step-by-Step Usage

1. **Upload Documents**:
   - In the left sidebar under **Upload Documents**, click or drag-and-drop one or multiple `.pdf`, `.docx`, or `.txt` files.
2. **Index Documents**:
   - Click the **⚡ Process & Index Documents** button.
   - The application will extract text, split it into chunks, generate sentence embeddings, and build the FAISS index.
   - The **Document Index Statistics** box will show the number of files and total chunks indexed.
3. **Ask Questions**:
   - Type your question in the chat input at the bottom and click **Ask AI 🚀** or press **Enter**.
4. **Inspect Sources**:
   - Expand the **🔍 Retrieved Sources & Context Inspector** accordion to see the exact text snippets and similarity scores that were retrieved from your documents to answer the question.
5. **Clear or Reset**:
   - Use **🧹 Clear Chat History** to wipe the conversation while keeping indexed documents.
   - Use **🗑️ Clear & Reset All** to wipe indexed documents, vector store, and chat history.

---

## 📁 Project Structure

```
Ask_My_Document/
├── app.py                 # Gradio Web Interface & UI Event Handlers
├── rag_engine.py          # Modular RAG functions (Load, Chunk, Embed, FAISS, OpenRouter)
├── test_rag.py            # Local verification & automated pipeline test script
├── sample_docs/           # Sample test documents (PDF, DOCX, TXT)
├── requirements.txt       # Project dependencies
├── .env.example           # Environment variable template
├── .env                   # Local secrets (ignored in Git)
├── .gitignore             # Git ignore configuration
└── README.md              # Documentation & User Guide
```

---

## ⚙️ Modular Functions in `rag_engine.py`

| Function | Description |
| :--- | :--- |
| `load_document(file_path)` | Extracts text and metadata from PDF, DOCX, and TXT files. |
| `split_documents(documents, chunk_size, chunk_overlap)` | Splits long texts into overlapping chunks using `RecursiveCharacterTextSplitter`. |
| `get_embedding_model(model_name)` | Initializes Hugging Face `sentence-transformers/all-MiniLM-L6-v2`. |
| `create_vector_store(chunks, embedding_model)` | Encodes chunks into dense vectors and builds an in-memory `FAISS` index. |
| `retrieve_relevant_chunks(vector_store, query, top_k)` | Queries FAISS using cosine/L2 semantic similarity to fetch top-k relevant excerpts. |
| `generate_answer_with_openrouter(query, chunks, api_key, model)` | Sends grounded prompt to OpenRouter LLM and returns structured answer & citations. |

---

## 🛡️ Anti-Hallucination & Grounding

This RAG application enforces strict grounding rules in the system prompt:
1. The LLM is instructed to answer **only** using facts present in the retrieved chunks.
2. If the user asks a question about information not present in the uploaded files, the system responds with:
   > *"I'm sorry, but based on the provided documents, I could not find information to answer this question."*
3. Every answer includes source tags and a collapsible Context Inspector showing the exact chunks retrieved.

---

## 🌐 Supported OpenRouter Models

You can use any model available on [OpenRouter](https://openrouter.ai/models), including:
- `openai/gpt-4o-mini` *(Recommended: Fast & cheap)*
- `google/gemini-2.0-flash-001` *(Recommended: Super-fast & capable)*
- `meta-llama/llama-3.3-70b-instruct:free` *(Free tier available on OpenRouter)*
- `anthropic/claude-3.5-haiku`
- `mistralai/mistral-7b-instruct`
- `deepseek/deepseek-chat`

---

## ☁️ Deployment Guides

### 1. 🚀 Deploy to Hugging Face Spaces (Recommended for Gradio)

Hugging Face Spaces provides **free, always-on hosting with 16 GB RAM and 2 vCPUs** specifically optimized for Gradio AI applications.

1. Create a free account on [huggingface.co](https://huggingface.co).
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
3. Fill in the Space settings:
   - **Space name**: `ask-my-documents`
   - **License**: `mit` / `apache-2.0`
   - **Space SDK**: **Gradio**
   - **Space hardware**: **Free - 2 vCPU · 16 GB · 50 GB disk**
4. Choose **Public** (or **Private**) and click **Create Space**.
5. Connect your GitHub repository or push your code via Git:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ask-my-documents
   git push space main
   ```
6. Add your OpenRouter API Key:
   - In your Space, go to **Settings** > **Variables and secrets**.
   - Under **Secrets**, click **New secret**.
   - Key: `OPENROUTER_API_KEY`
   - Value: `your_openrouter_api_key`
7. Hugging Face will automatically build and launch your Space within 1-2 minutes!

---

### 2. ⚡ Deploy to Render

1. Push your repository to **GitHub**.
2. Log into [Render.com](https://render.com) and click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Configure service settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. In **Environment Variables**, add:
   - `OPENROUTER_API_KEY`: `your_openrouter_api_key`
   - `OPENROUTER_MODEL`: `openai/gpt-4o-mini` (optional)
   - `EMBEDDING_MODEL_NAME`: `sentence-transformers/all-MiniLM-L6-v2` (optional)
6. Click **Create Web Service**! Render will deploy your RAG assistant to a public URL.


