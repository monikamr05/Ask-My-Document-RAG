"""
RAG Engine - Document Processing, Embedding, FAISS Indexing, and OpenRouter LLM Answering.
========================================================================================
This module contains modular, beginner-friendly functions for:
1. Loading documents (PDF, DOCX, TXT)
2. Splitting text into chunks
3. Embedding text with Hugging Face sentence-transformers & indexing with FAISS
4. Retrieving relevant chunks based on semantic similarity
5. Generating grounded answers via OpenRouter API
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain Document model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding & Vector Store
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

# Document Parsers
try:
    import pymupdf  # Ultra-fast C++ based PDF engine
except ImportError:
    pymupdf = None

from pypdf import PdfReader
import docx

# LLM Client (OpenRouter uses the OpenAI-compatible API)
from openai import OpenAI


# ==========================================
# 1. EMBEDDING MODEL INITIALIZER
# ==========================================
def get_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Initializes and returns the Hugging Face Sentence Transformers embedding model.
    Default model: 'sentence-transformers/all-MiniLM-L6-v2' (fast, lightweight, high quality).
    """
    if not model_name:
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Hugging Face embeddings with optimized batch processing
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
    )
    return embeddings


# ==========================================
# 2. DOCUMENT LOADING FUNCTION (HIGH SPEED)
# ==========================================
def load_document(file_path: str, original_filename: Optional[str] = None) -> List[Document]:
    """
    Loads text from PDF, DOCX, or TXT files and returns a list of LangChain Document objects.
    Uses high-speed PyMuPDF for PDFs with fallback to pypdf.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = original_filename or os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    if not extension:
        extension = os.path.splitext(file_path)[1].lower()
        
    documents: List[Document] = []

    # 1. Handle PDF Documents (Fast PyMuPDF engine)
    if extension == ".pdf":
        if pymupdf is not None:
            try:
                pdf_doc = pymupdf.open(file_path)
                total_pages = len(pdf_doc)
                for page_idx in range(total_pages):
                    page = pdf_doc[page_idx]
                    text = page.get_text("text").strip()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": page_idx + 1,
                                    "total_pages": total_pages,
                                    "file_type": "PDF"
                                }
                            )
                        )
                pdf_doc.close()
            except Exception:
                documents = []  # Fallback to pypdf below
        
        # Fallback to pypdf if pymupdf didn't produce docs
        if not documents:
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": filename,
                                    "page": page_idx + 1,
                                    "total_pages": total_pages,
                                    "file_type": "PDF"
                                }
                            )
                        )
            except Exception as e:
                raise ValueError(f"Could not extract text from PDF '{filename}': {str(e)}")

    # 2. Handle DOCX Documents
    elif extension == ".docx":
        try:
            doc = docx.Document(file_path)
            paragraphs_text = []
            
            # Extract normal paragraphs
            for p in doc.paragraphs:
                clean_p = p.text.strip()
                if clean_p:
                    paragraphs_text.append(clean_p)
            
            # Extract tables if present
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        paragraphs_text.append(" | ".join(row_cells))
            
            full_text = "\n\n".join(paragraphs_text)
            if full_text.strip():
                documents.append(
                    Document(
                        page_content=full_text,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "total_pages": 1,
                            "file_type": "DOCX"
                        }
                    )
                )
        except Exception as e:
            raise ValueError(f"Could not read DOCX '{filename}': {str(e)}")

    # 3. Handle Plain TXT Documents
    elif extension == ".txt":
        content = ""
        # Try multiple encodings for resilience
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if not content:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                content = ""

        content = content.strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "page": 1,
                        "total_pages": 1,
                        "file_type": "TXT"
                    }
                )
            )
    else:
        raise ValueError(f"Unsupported file format '{extension}'. Please upload .pdf, .docx, or .txt files.")

    if not documents:
        raise ValueError(f"No readable text could be extracted from '{filename}'.")

    return documents


# ==========================================
# 3. TEXT CHUNKING FUNCTION
# ==========================================
def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits long documents into smaller overlapping chunks suitable for semantic search and LLM context limits.
    Maintains source metadata and assigns unique chunk IDs.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add sequential chunk IDs to metadata for easy citation reference
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
        
    return chunks


# ==========================================
# 4. VECTOR STORE & FAISS INDEXING
# ==========================================
def create_vector_store(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None
) -> FAISS:
    """
    Generates embeddings for each chunk and builds an in-memory FAISS vector index.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty chunk list.")
    
    if embedding_model is None:
        embedding_model = get_embedding_model()
        
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


# ==========================================
# 5. SIMILARITY RETRIEVAL FUNCTION
# ==========================================
def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4
) -> List[Tuple[Document, float]]:
    """
    Queries the FAISS vector database to retrieve the top_k most semantically relevant chunks.
    Returns a list of tuples: (Document, similarity_distance_score).
    """
    if vector_store is None:
        raise ValueError("Vector database is empty. Please upload and process documents first.")
    
    query_str = (query or "").strip()
    if not query_str:
        return []
        
    # similarity_search_with_score returns L2 distance (lower score = higher similarity)
    results = vector_store.similarity_search_with_score(query_str, k=top_k)
    return results


# ==========================================
# 6. OPENROUTER LLM ANSWER GENERATION
# ==========================================
def generate_answer_with_openrouter(
    query: str,
    retrieved_chunks: List[Tuple[Document, float]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a strictly grounded answer from OpenRouter LLM based solely on retrieved document chunks.
    If the answer cannot be found in the context, clearly states that information is unavailable.
    
    Returns a dictionary containing:
    - 'answer': The LLM generated response
    - 'sources': Structured list of sources and snippets
    - 'model': The model used
    """
    # 1. Resolve Model Name
    resolved_model = (model_name or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")).strip()

    # 2. Check if retrieved chunks exist
    if not retrieved_chunks:
        return {
            "answer": "I'm sorry, but based on the provided documents, I could not find information to answer this question.",
            "sources": [],
            "model": resolved_model
        }

    # 3. Resolve API Key
    resolved_api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
    if not resolved_api_key:
        raise ValueError(
            "OpenRouter API Key not found! Please provide it in the UI settings or set OPENROUTER_API_KEY in your .env file."
        )

    # 4. Format Context from Chunks
    formatted_context_list = []
    sources_summary = []

    for idx, (doc, score) in enumerate(retrieved_chunks):
        src_name = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", idx + 1)
        content_snippet = doc.page_content.strip()

        # Context block for prompt
        context_block = (
            f"[Source {idx + 1}: {src_name} | Page {page_num} | Chunk #{chunk_id}]\n"
            f"{content_snippet}"
        )
        formatted_context_list.append(context_block)

        # Structured source summary for UI display
        sources_summary.append({
            "index": idx + 1,
            "source": src_name,
            "page": page_num,
            "chunk_id": chunk_id,
            "snippet": content_snippet,
            "score": round(float(score), 4) if score is not None else None
        })

    full_context_text = "\n\n---------------------\n\n".join(formatted_context_list)

    # 5. Strict Grounding System Prompt
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
        f"{full_context_text}\n"
        f"==============================\n\n"
        f"User Question: {query}\n\n"
        f"Please provide an accurate answer based strictly on the above context:"
    )

    # 6. Call OpenRouter API (OpenAI Compatible)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=resolved_api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Ask My Documents - AI RAG Assistant",
        }
    )

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for factual precision
            max_tokens=1000
        )
        answer_text = ""
        if response and response.choices and len(response.choices) > 0:
            msg = response.choices[0].message
            if msg and msg.content:
                answer_text = msg.content.strip()
        
        if not answer_text:
            answer_text = "I'm sorry, but based on the provided documents, I could not find information to answer this question."
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise ValueError("Invalid OpenRouter API Key. Please check your key at https://openrouter.ai/keys.")
        elif "402" in error_msg or "Payment" in error_msg or "credits" in error_msg.lower():
            raise ValueError("OpenRouter account credit limit reached. Please check your balance or use a free model (e.g. meta-llama/llama-3.3-70b-instruct:free).")
        else:
            raise RuntimeError(f"OpenRouter API error: {error_msg}")

    return {
        "answer": answer_text,
        "sources": sources_summary,
        "model": resolved_model
    }
