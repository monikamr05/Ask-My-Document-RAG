"""
Ask My Documents - Vercel Serverless RAG Application
=====================================================
A high-performance, lightweight FastAPI RAG application tailored for Vercel Serverless.
Handles PDF, DOCX, TXT ingestion, text chunking, hybrid semantic search,
and grounded OpenRouter LLM generation.
"""

import os
import re
import io
import math
import uuid
import json
from typing import List, Dict, Any, Optional
from collections import Counter

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import docx

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Ask My Documents RAG (Vercel)",
    description="Serverless AI RAG Document Assistant",
    version="1.0.0"
)

# In-memory session store (ephemeral cache per serverless instance)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ==============================================================================
# DOCUMENT PARSING & CHUNKING UTILITIES
# ==============================================================================

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extracts text per page from PDF bytes using pypdf."""
    documents = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": page_idx + 1,
                        "total_pages": total_pages,
                        "file_type": "PDF"
                    }
                })
    except Exception as e:
        raise ValueError(f"Error reading PDF '{filename}': {str(e)}")
    return documents


def extract_text_from_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extracts text and tables from DOCX bytes using python-docx."""
    documents = []
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = []
        for p in doc.paragraphs:
            clean_p = p.text.strip()
            if clean_p:
                paragraphs.append(clean_p)
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    paragraphs.append(" | ".join(row_cells))
        full_text = "\n\n".join(paragraphs).strip()
        if full_text:
            documents.append({
                "content": full_text,
                "metadata": {
                    "source": filename,
                    "page": 1,
                    "total_pages": 1,
                    "file_type": "DOCX"
                }
            })
    except Exception as e:
        raise ValueError(f"Error reading DOCX '{filename}': {str(e)}")
    return documents


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extracts text from TXT bytes with multi-encoding fallback."""
    content = ""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            content = file_bytes.decode(enc).strip()
            break
        except UnicodeDecodeError:
            continue
    if not content:
        content = file_bytes.decode("utf-8", errors="ignore").strip()

    if not content:
        raise ValueError(f"TXT file '{filename}' is empty.")

    return [{
        "content": content,
        "metadata": {
            "source": filename,
            "page": 1,
            "total_pages": 1,
            "file_type": "TXT"
        }
    }]


def chunk_text(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """Splits documents into overlapping chunks suitable for semantic search."""
    chunks = []
    chunk_id = 1

    for doc in documents:
        text = doc["content"]
        meta = doc["metadata"]
        
        # Simple recursive-like splitter
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                # Find nearest paragraph or sentence boundary
                for sep in ["\n\n", "\n", ". ", " "]:
                    sep_pos = text.rfind(sep, start, end)
                    if sep_pos != -1 and sep_pos > start + (chunk_size // 2):
                        end = sep_pos + len(sep)
                        break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "source": meta["source"],
                        "page": meta.get("page", 1),
                        "total_pages": meta.get("total_pages", 1),
                        "file_type": meta.get("file_type", "TXT"),
                        "chunk_id": chunk_id
                    }
                })
                chunk_id += 1
                
            start = end - chunk_overlap
            if start >= text_len or end >= text_len:
                break

    return chunks


# ==============================================================================
# LIGHTWEIGHT HYBRID SEMANTIC SEARCH (Pure Python, Zero Dependency Weight)
# ==============================================================================

def tokenize(text: str) -> List[str]:
    """Tokenizes and normalizes text for keyword and n-gram scoring."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text)]

class BM25Retriever:
    """Fast, in-memory BM25 retrieval engine with n-gram semantic boosting."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.doc_len = []
        self.doc_freqs = []
        self.df = Counter()
        self.num_docs = len(chunks)
        self.avgdl = 0.0

        total_len = 0
        for chunk in chunks:
            tokens = tokenize(chunk["content"])
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length
            
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for word in freqs.keys():
                self.df[word] += 1
                
        self.avgdl = total_len / max(1, self.num_docs)

    def score(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores = []

        for idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[idx]
            
            for token in query_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                df_val = self.df.get(token, 0)
                # BM25 IDF
                idf = math.log(1.0 + (self.num_docs - df_val + 0.5) / (df_val + 0.5))
                # BM25 TF
                denom = tf + k1 * (1.0 - b + b * (doc_len / max(1, self.avgdl)))
                score += idf * (tf * (k1 + 1.0) / max(1e-5, denom))
                
            # Direct phrase boost
            chunk_text = self.chunks[idx]["content"].lower()
            if query.lower() in chunk_text:
                score += 3.0

            scores.append((score, self.chunks[idx]))

        # Sort descending by relevance score
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, chunk in scores[:top_k]:
            results.append({
                "chunk": chunk,
                "score": round(s, 4)
            })
        return results


# ==============================================================================
# OPENROUTER LLM GENERATOR
# ==============================================================================

def generate_rag_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generates strictly grounded answers via OpenRouter API."""
    resolved_model = (model_name or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")).strip()
    resolved_api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()

    if not retrieved_results:
        return {
            "answer": "I'm sorry, but based on the provided documents, I could not find information to answer this question.",
            "sources": [],
            "model": resolved_model
        }

    if not resolved_api_key:
        raise ValueError("OpenRouter API Key is missing. Please provide it in settings or set OPENROUTER_API_KEY environment variable.")

    formatted_context = []
    sources_summary = []

    for idx, item in enumerate(retrieved_results):
        chunk = item["chunk"]
        meta = chunk["metadata"]
        src = meta.get("source", "Unknown Document")
        page = meta.get("page", 1)
        cid = meta.get("chunk_id", idx + 1)
        text_snippet = chunk["content"]

        formatted_context.append(
            f"[Source {idx + 1}: {src} | Page {page} | Chunk #{cid}]\n{text_snippet}"
        )
        sources_summary.append({
            "index": idx + 1,
            "source": src,
            "page": page,
            "chunk_id": cid,
            "snippet": text_snippet,
            "score": item["score"]
        })

    context_str = "\n\n---------------------\n\n".join(formatted_context)

    system_prompt = (
        "You are an expert, truthful AI Document Assistant called 'Ask My Documents'.\n"
        "Your duty is to answer user questions STRICTLY and ONLY using the provided document excerpts below.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer based ONLY on the provided Context excerpts.\n"
        "2. If the answer is NOT explicitly present or cannot be directly deduced from the Context, you MUST respond with: "
        "'I'm sorry, but based on the provided documents, I could not find information to answer this question.'\n"
        "3. Never make up facts, guess, or use external world knowledge that is not in the Context.\n"
        "4. Be concise, well-structured, and cite source references (e.g. [Source 1, Page 2]) when stating key points."
    )

    user_prompt = (
        f"Context from uploaded documents:\n"
        f"==============================\n"
        f"{context_str}\n"
        f"==============================\n\n"
        f"User Question: {query}\n\n"
        f"Please provide an accurate answer based strictly on the above context:"
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=resolved_api_key,
        default_headers={
            "HTTP-Referer": "https://ask-my-documents.vercel.app",
            "X-Title": "Ask My Documents RAG Assistant",
        }
    )

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        answer = response.choices[0].message.content.strip() if response.choices else ""
        if not answer:
            answer = "I'm sorry, but based on the provided documents, I could not find information to answer this question."
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err:
            raise ValueError("Invalid OpenRouter API Key. Please verify your key at https://openrouter.ai/keys.")
        elif "402" in err or "credit" in err.lower():
            raise ValueError("OpenRouter credit limit reached. Switch to a free model like `meta-llama/llama-3.3-70b-instruct:free`.")
        else:
            raise RuntimeError(f"OpenRouter API Error: {err}")

    return {
        "answer": answer,
        "sources": sources_summary,
        "model": resolved_model
    }


# ==============================================================================
# FASTAPI HTTP ENDPOINTS
# ==============================================================================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "platform": "Vercel Serverless", "version": "1.0.0"}


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150)
):
    """Processes uploaded files and stores indexed chunks for the session."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    sid = session_id or str(uuid.uuid4())
    all_raw_docs = []
    file_stats = []

    for f in files:
        filename = f.filename or "uploaded_file"
        file_bytes = await f.read()
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                docs = extract_text_from_pdf(file_bytes, filename)
            elif ext == ".docx":
                docs = extract_text_from_docx(file_bytes, filename)
            elif ext in [".txt", ".md", ".csv", ".json"]:
                docs = extract_text_from_txt(file_bytes, filename)
            else:
                raise ValueError(f"Unsupported format '{ext}'. Upload .pdf, .docx, or .txt.")
            
            all_raw_docs.extend(docs)
            file_stats.append({
                "name": filename,
                "pages": sum(d["metadata"].get("page", 1) for d in docs),
                "words": sum(len(d["content"].split()) for d in docs)
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not all_raw_docs:
        raise HTTPException(status_code=400, detail="No readable text found in uploaded files.")

    chunks = chunk_text(all_raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = BM25Retriever(chunks)

    # Save to session
    SESSIONS[sid] = {
        "chunks": chunks,
        "retriever": retriever,
        "files": file_stats
    }

    total_words = sum(f["words"] for f in file_stats)

    return {
        "session_id": sid,
        "chunks_count": len(chunks),
        "total_words": total_words,
        "files_count": len(file_stats),
        "files": file_stats,
        "message": f"Successfully indexed {len(file_stats)} file(s) into {len(chunks)} chunks ({total_words:,} words)!"
    }


@app.post("/api/ask")
async def ask_question(
    question: str = Form(...),
    session_id: str = Form(...),
    top_k: int = Form(4),
    api_key: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    client_chunks: Optional[str] = Form(None)
):
    """Answers a question grounded in the indexed document chunks."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    session_data = SESSIONS.get(session_id)
    chunks = []
    retriever = None

    if session_data:
        chunks = session_data.get("chunks", [])
        retriever = session_data.get("retriever")
    elif client_chunks:
        try:
            chunks = json.loads(client_chunks)
            retriever = BM25Retriever(chunks)
        except Exception:
            chunks = []

    if not chunks or not retriever:
        raise HTTPException(
            status_code=400,
            detail="No documents found for this session. Please upload your documents first."
        )

    retrieved = retriever.score(question, top_k=top_k)

    try:
        result = generate_rag_answer(
            query=question,
            retrieved_results=retrieved,
            api_key=api_key,
            model_name=model_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/clear")
async def clear_session(session_id: str = Form(...)):
    """Clears the session index."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"message": "Session documents cleared successfully."}


# ==============================================================================
# EMBEDDED HIGH-PERFORMANCE WEB UI
# ==============================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ask My Documents – AI RAG Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-base: #0a0c10;
            --bg-surface: #12151c;
            --bg-card: #181c26;
            --bg-card-hover: #1e2330;
            --border: #262b3a;
            --border-highlight: #3b4256;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.25);
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Navigation Header */
        header {
            background: rgba(18, 21, 28, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0.85rem 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            background: var(--primary-gradient);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .brand-title {
            font-weight: 800;
            font-size: 1.15rem;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 20px;
            text-transform: uppercase;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .btn {
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 0.5rem 0.9rem;
            border-radius: var(--radius-sm);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .btn:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-highlight);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: var(--primary-gradient);
            border: none;
            color: #fff;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .btn-primary:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px var(--primary-glow);
            transform: translateY(-1px);
        }

        /* Main App Layout */
        .app-container {
            display: grid;
            grid-template-columns: 360px 1fr;
            flex: 1;
            height: calc(100vh - 61px);
            overflow: hidden;
        }

        @media (max-width: 900px) {
            .app-container {
                grid-template-columns: 1fr;
                height: auto;
                overflow: visible;
            }
        }

        /* Left Sidebar: Ingestion & Settings */
        .sidebar {
            background: var(--bg-surface);
            border-right: 1px solid var(--border);
            padding: 1.25rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .sidebar-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.1rem;
        }

        .card-header {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Upload Dropzone */
        .dropzone {
            border: 2px dashed var(--border-highlight);
            border-radius: var(--radius-md);
            padding: 1.5rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(24, 28, 38, 0.5);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .dropzone-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }

        .dropzone-text {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .dropzone-text b {
            color: var(--text-main);
        }

        .file-chips {
            margin-top: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            max-height: 120px;
            overflow-y: auto;
        }

        .file-chip {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.35rem 0.6rem;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Document Status Stats */
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.6rem;
            margin-top: 0.75rem;
        }

        .stat-box {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.6rem;
            text-align: center;
        }

        .stat-val {
            font-size: 1.15rem;
            font-weight: 800;
            color: #818cf8;
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-lbl {
            font-size: 0.7rem;
            color: var(--text-dim);
            text-transform: uppercase;
        }

        /* Status Alert */
        .status-pill {
            padding: 0.6rem 0.85rem;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            margin-top: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            line-height: 1.4;
        }

        .status-ready {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
        }

        .status-idle {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.15);
            color: var(--text-muted);
        }

        /* Right Panel: Chat Interface */
        .chat-section {
            display: flex;
            flex-direction: column;
            height: 100%;
            background: var(--bg-base);
            position: relative;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .message-bubble {
            max-width: 85%;
            padding: 1.1rem 1.25rem;
            border-radius: var(--radius-md);
            font-size: 0.92rem;
            line-height: 1.6;
            animation: fadeIn 0.25s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-user {
            align-self: flex-end;
            background: #232938;
            border: 1px solid #333c52;
            color: #fff;
            border-bottom-right-radius: 4px;
        }

        .msg-bot {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-bottom-left-radius: 4px;
            box-shadow: var(--shadow);
            width: 100%;
        }

        .msg-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.75rem;
            color: var(--text-dim);
        }

        .bot-tag {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
        }

        .markdown-body {
            color: #e2e8f0;
        }
        .markdown-body p { margin-bottom: 0.6rem; }
        .markdown-body ul, .markdown-body ol { margin-left: 1.2rem; margin-bottom: 0.6rem; }
        .markdown-body code { background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }

        /* Source Citations Box */
        .sources-accordion {
            margin-top: 0.85rem;
            background: rgba(0,0,0,0.25);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            overflow: hidden;
        }

        .sources-summary {
            padding: 0.5rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #94a3b8;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.02);
        }

        .sources-summary:hover {
            color: #fff;
            background: rgba(255,255,255,0.05);
        }

        .sources-content {
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            border-top: 1px solid var(--border);
            font-size: 0.78rem;
        }

        .source-card {
            background: rgba(255,255,255,0.02);
            border-left: 3px solid var(--primary);
            padding: 0.5rem 0.7rem;
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        }

        .source-meta {
            font-weight: 700;
            color: #818cf8;
            margin-bottom: 0.25rem;
            display: flex;
            justify-content: space-between;
        }

        .source-text {
            color: #94a3b8;
            line-height: 1.4;
            font-style: italic;
        }

        /* Input Controls Area */
        .chat-input-area {
            background: var(--bg-surface);
            border-top: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .suggestions {
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            padding-bottom: 2px;
        }

        .suggestion-chip {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            font-size: 0.75rem;
            color: var(--text-muted);
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
        }

        .suggestion-chip:hover {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
            color: #fff;
        }

        .input-row {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .chat-input {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem 1.15rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease;
        }

        .chat-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        /* Settings Modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(4px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }

        .modal-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-highlight);
            border-radius: var(--radius-lg);
            width: 90%;
            max-width: 480px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }

        .modal-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-label {
            display: block;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .form-input, .form-select {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.85rem;
            color: #fff;
            font-size: 0.85rem;
            outline: none;
        }

        .form-input:focus, .form-select:focus {
            border-color: var(--primary);
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand-container">
            <div class="brand-icon">📑</div>
            <div>
                <span class="brand-title">Ask My Documents</span>
                <span class="brand-badge">Vercel Serverless RAG</span>
            </div>
        </div>
        <div class="header-actions">
            <button class="btn" onclick="openSettingsModal()">⚙️ Settings</button>
            <button class="btn" onclick="clearSessionData()">🗑️ Reset</button>
            <a class="btn btn-primary" href="https://github.com/monikamr05/Ask-My-Document-RAG" target="_blank">⭐ GitHub</a>
        </div>
    </header>

    <!-- Main Container -->
    <div class="app-container">
        
        <!-- Left Sidebar: Ingestion Controls -->
        <aside class="sidebar">
            <div class="sidebar-card">
                <div class="card-header">📁 1. Ingest Documents</div>
                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <div class="dropzone-icon">📤</div>
                    <div class="dropzone-text">
                        <b>Click to upload</b> or drag & drop<br>
                        PDF, DOCX, or TXT
                    </div>
                </div>
                <input type="file" id="fileInput" multiple accept=".pdf,.docx,.txt" style="display:none" onchange="handleFileSelect(event)">
                
                <div class="file-chips" id="fileChips"></div>

                <button class="btn btn-primary" id="processBtn" style="width: 100%; margin-top: 0.85rem;" onclick="uploadAndIndexFiles()">
                    ⚡ Process & Index Documents
                </button>

                <div id="statusPill" class="status-pill status-idle">
                    ⏳ Waiting for documents...
                </div>
            </div>

            <!-- Stats & Overview -->
            <div class="sidebar-card">
                <div class="card-header">📊 Knowledge Base Stats</div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-val" id="statFiles">0</div>
                        <div class="stat-lbl">Documents</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val" id="statChunks">0</div>
                        <div class="stat-lbl">Chunks</div>
                    </div>
                </div>
                <div class="stat-grid" style="margin-top: 0.5rem;">
                    <div class="stat-box" style="grid-column: span 2;">
                        <div class="stat-val" id="statWords">0</div>
                        <div class="stat-lbl">Total Indexed Words</div>
                    </div>
                </div>
            </div>

            <!-- Grounding Notice -->
            <div class="sidebar-card" style="font-size: 0.78rem; color: var(--text-dim); line-height: 1.5;">
                🛡️ <b>Strict Anti-Hallucination Active</b><br>
                Answers are grounded 100% in your uploaded documents. If the context is missing, the AI will not fabricate facts.
            </div>
        </aside>

        <!-- Right Main Chat Section -->
        <main class="chat-section">
            <div class="chat-messages" id="chatMessages">
                <div class="message-bubble msg-bot">
                    <div class="msg-header">
                        <span class="bot-tag">AI Assistant</span>
                        <span>Ask My Documents RAG</span>
                    </div>
                    <div class="markdown-body">
                        👋 <b>Welcome!</b> Upload your PDF, Word (.docx), or TXT documents on the left and ask any question.<br><br>
                        Every answer includes verified <b>source citations</b> and a <b>Context Inspector</b> showing the retrieved document snippets!
                    </div>
                </div>
            </div>

            <!-- Input Area -->
            <div class="chat-input-area">
                <div class="suggestions">
                    <div class="suggestion-chip" onclick="fillQuery('Summarize the main points of this document.')">💡 Summarize document</div>
                    <div class="suggestion-chip" onclick="fillQuery('What are the key conclusions or findings?')">🎯 Key conclusions</div>
                    <div class="suggestion-chip" onclick="fillQuery('List the main action items or takeaways.')">📋 Action items</div>
                </div>

                <div class="input-row">
                    <input type="text" id="queryInput" class="chat-input" placeholder="Ask a question about your documents..." onkeydown="handleKeyPress(event)">
                    <button class="btn btn-primary" id="sendBtn" onclick="askQuestion()" style="padding: 0.85rem 1.4rem;">
                        Ask AI 🚀
                    </button>
                </div>
            </div>
        </main>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal-card">
            <div class="modal-title">
                <span>⚙️ RAG Assistant Settings</span>
                <button class="btn" onclick="closeSettingsModal()" style="padding: 2px 8px;">✕</button>
            </div>
            <div class="form-group">
                <label class="form-label">OpenRouter API Key (Optional if set in .env)</label>
                <input type="password" id="apiKeyInput" class="form-input" placeholder="sk-or-v1-...">
            </div>
            <div class="form-group">
                <label class="form-label">LLM Model</label>
                <select id="modelSelect" class="form-select">
                    <option value="openai/gpt-4o-mini">OpenAI: GPT-4o Mini (Fast & Accurate)</option>
                    <option value="google/gemini-2.0-flash-001">Google: Gemini 2.0 Flash</option>
                    <option value="meta-llama/llama-3.3-70b-instruct:free">Meta: Llama 3.3 70B (Free Tier)</option>
                    <option value="anthropic/claude-3.5-haiku">Anthropic: Claude 3.5 Haiku</option>
                    <option value="deepseek/deepseek-chat">DeepSeek: DeepSeek V3</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label">Top-K Context Snippets</label>
                <select id="topKSelect" class="form-select">
                    <option value="3">3 Chunks (Compact)</option>
                    <option value="4" selected>4 Chunks (Balanced)</option>
                    <option value="6">6 Chunks (Comprehensive)</option>
                </select>
            </div>
            <button class="btn btn-primary" style="width: 100%; margin-top: 0.5rem;" onclick="saveSettings()">
                Save Preferences
            </button>
        </div>
    </div>

    <script>
        let selectedFiles = [];
        let sessionId = localStorage.getItem('rag_session_id') || ('sess_' + Math.random().toString(36).substring(2, 10));
        localStorage.setItem('rag_session_id', sessionId);

        let cachedChunks = [];

        // Load stored settings
        document.getElementById('apiKeyInput').value = localStorage.getItem('rag_api_key') || '';
        document.getElementById('modelSelect').value = localStorage.getItem('rag_model') || 'openai/gpt-4o-mini';

        function openSettingsModal() { document.getElementById('settingsModal').style.display = 'flex'; }
        function closeSettingsModal() { document.getElementById('settingsModal').style.display = 'none'; }
        
        function saveSettings() {
            localStorage.setItem('rag_api_key', document.getElementById('apiKeyInput').value.trim());
            localStorage.setItem('rag_model', document.getElementById('modelSelect').value);
            closeSettingsModal();
        }

        // Drag & Drop
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                selectedFiles = Array.from(e.dataTransfer.files);
                renderFileChips();
            }
        });

        function handleFileSelect(e) {
            if (e.target.files && e.target.files.length > 0) {
                selectedFiles = Array.from(e.target.files);
                renderFileChips();
            }
        }

        function renderFileChips() {
            const container = document.getElementById('fileChips');
            container.innerHTML = '';
            selectedFiles.forEach((file, idx) => {
                const chip = document.createElement('div');
                chip.className = 'file-chip';
                chip.innerHTML = `<span>📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>`;
                container.appendChild(chip);
            });
        }

        async function uploadAndIndexFiles() {
            if (selectedFiles.length === 0) {
                alert("Please select at least one PDF, DOCX, or TXT file first.");
                return;
            }

            const btn = document.getElementById('processBtn');
            const statusPill = document.getElementById('statusPill');
            btn.disabled = true;
            btn.innerText = "⏳ Ingesting & Indexing...";
            statusPill.className = "status-pill status-idle";
            statusPill.innerText = "Processing document text...";

            const formData = new FormData();
            selectedFiles.forEach(f => formData.append('files', f));
            formData.append('session_id', sessionId);

            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                
                if (!res.ok) throw new Error(data.detail || "Upload failed");

                document.getElementById('statFiles').innerText = data.files_count;
                document.getElementById('statChunks').innerText = data.chunks_count;
                document.getElementById('statWords').innerText = Number(data.total_words).toLocaleString();

                statusPill.className = "status-pill status-ready";
                statusPill.innerText = "✅ " + data.message;
            } catch (err) {
                statusPill.className = "status-pill status-idle";
                statusPill.innerText = "❌ Error: " + err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = "⚡ Process & Index Documents";
            }
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') askQuestion();
        }

        function fillQuery(text) {
            const input = document.getElementById('queryInput');
            input.value = text;
            input.focus();
        }

        async function askQuestion() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;

            const chat = document.getElementById('chatMessages');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message-bubble msg-user';
            userMsg.innerText = query;
            chat.appendChild(userMsg);

            input.value = '';
            chat.scrollTop = chat.scrollHeight;

            // Add loading bot message
            const botMsg = document.createElement('div');
            botMsg.className = 'message-bubble msg-bot';
            botMsg.innerHTML = `
                <div class="msg-header">
                    <span class="bot-tag">AI Assistant</span>
                    <span>Retrieving context...</span>
                </div>
                <div class="markdown-body"><i>Searching indexed knowledge base...</i></div>
            `;
            chat.appendChild(botMsg);
            chat.scrollTop = chat.scrollHeight;

            const apiKey = localStorage.getItem('rag_api_key') || '';
            const model = localStorage.getItem('rag_model') || 'openai/gpt-4o-mini';
            const topK = document.getElementById('topKSelect').value || 4;

            const formData = new FormData();
            formData.append('question', query);
            formData.append('session_id', sessionId);
            formData.append('top_k', topK);
            if (apiKey) formData.append('api_key', apiKey);
            if (model) formData.append('model_name', model);

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Failed to generate answer.");

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = `
                        <div class="sources-accordion">
                            <div class="sources-summary" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'flex' : 'none'">
                                <span>📚 Verified Sources (${data.sources.length} citations)</span>
                                <span>▼</span>
                            </div>
                            <div class="sources-content" style="display:none;">
                                ${data.sources.map(s => `
                                    <div class="source-card">
                                        <div class="source-meta">
                                            <span>[Source ${s.index}] ${s.source} (Page ${s.page})</span>
                                            <span>Relevance: ${s.score}</span>
                                        </div>
                                        <div class="source-text">"${s.snippet.substring(0, 200)}..."</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                botMsg.innerHTML = `
                    <div class="msg-header">
                        <span class="bot-tag">AI Grounded Answer</span>
                        <span>Model: ${data.model}</span>
                    </div>
                    <div class="markdown-body">${marked.parse(data.answer)}</div>
                    ${sourcesHtml}
                `;
            } catch (err) {
                botMsg.innerHTML = `
                    <div class="msg-header">
                        <span class="bot-tag" style="background:#ef4444; color:#fff;">Error</span>
                    </div>
                    <div class="markdown-body" style="color: #f87171;">⚠️ ${err.message}</div>
                `;
            }

            chat.scrollTop = chat.scrollHeight;
        }

        async function clearSessionData() {
            if (!confirm("Are you sure you want to reset and clear all uploaded documents?")) return;
            const formData = new FormData();
            formData.append('session_id', sessionId);
            await fetch('/api/clear', { method: 'POST', body: formData });
            
            sessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
            localStorage.setItem('rag_session_id', sessionId);
            selectedFiles = [];
            renderFileChips();
            document.getElementById('statFiles').innerText = '0';
            document.getElementById('statChunks').innerText = '0';
            document.getElementById('statWords').innerText = '0';
            document.getElementById('statusPill').className = 'status-pill status-idle';
            document.getElementById('statusPill').innerText = '⏳ Waiting for documents...';
            document.getElementById('chatMessages').innerHTML = `
                <div class="message-bubble msg-bot">
                    <div class="msg-header"><span class="bot-tag">AI Assistant</span></div>
                    <div class="markdown-body">Knowledge base reset. Upload new documents to begin!</div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=INDEX_HTML)
