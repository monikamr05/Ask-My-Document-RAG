"""
Ask My Documents – AI RAG Assistant
====================================
A high-performance, professional Retrieval-Augmented Generation (RAG) web application
built with Gradio, LangChain, FAISS, PyMuPDF, Hugging Face sentence-transformers, and OpenRouter API.
"""

import os
import threading
import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional

# Import modular RAG engine functions
from rag_engine import (
    get_embedding_model,
    load_document,
    split_documents,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_answer_with_openrouter,
)

# Load environment variables from .env file
load_dotenv()

# Global embedding model cache for instant sub-second indexing
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL = None

def get_or_load_embedding_model():
    """Initializes and caches the embedding model for ultra-fast document indexing."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = get_embedding_model(EMBEDDING_MODEL_NAME)
    return EMBEDDING_MODEL

# Pre-load the embedding model synchronously so it is 100% warm in memory for sub-second uploads
print("[INFO] Pre-loading embedding model into memory for instantaneous indexing (< 2s)...")
try:
    EMBEDDING_MODEL = get_or_load_embedding_model()
    print("[INFO] [OK] Embedding model pre-loaded and ready in memory!")
except Exception as e:
    print(f"[Warning] Synchronous model preload fallback: {e}")


# ==============================================================================
# HIGH-PERFORMANCE WORKFLOW FUNCTIONS
# ==============================================================================

def extract_file_info(file_obj: Any) -> Tuple[str, str]:
    """Safely extracts filesystem path and original filename across Gradio versions."""
    if isinstance(file_obj, str):
        return file_obj, os.path.basename(file_obj)
    elif isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name") or ""
        name = file_obj.get("orig_name") or os.path.basename(path)
        return path, name
    else:
        path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None) or str(file_obj)
        name = getattr(file_obj, "orig_name", None) or os.path.basename(path)
        return path, name


def process_uploaded_files(
    file_objs: List[Any],
    chunk_size: int,
    chunk_overlap: int,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Any, str, str, str]:
    """
    High-speed document processor:
    - Uses PyMuPDF (C++ backend) for sub-second PDF text extraction
    - Batch chunking with LangChain
    - Pre-warmed FAISS vector store indexing
    """
    if not file_objs:
        return (
            None,
            "<div class='status-pill status-warning'>⚠️ Please select at least one PDF, DOCX, or TXT file.</div>",
            "_No documents indexed yet._",
            "Upload documents to inspect retrieved context snippets here."
        )

    all_documents = []
    file_names = []
    
    progress(0.15, desc="Reading documents with high-speed parser...")
    
    for file_obj in file_objs:
        file_path, file_name = extract_file_info(file_obj)
        if not file_path:
            continue
        file_names.append(file_name)
        
        try:
            docs = load_document(file_path, original_filename=file_name)
            all_documents.extend(docs)
        except Exception as e:
            return (
                None,
                f"<div class='status-pill status-danger'>❌ Error reading <b>{file_name}</b>: {str(e)}</div>",
                "_Indexing failed._",
                "No context available."
            )

    if not all_documents:
        return (
            None,
            "<div class='status-pill status-danger'>❌ No readable text found in uploaded files.</div>",
            "_No documents indexed._",
            "No context available."
        )

    # Step 2: Split text into chunks
    progress(0.45, desc="Generating semantic chunks...")
    try:
        chunks = split_documents(all_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        return (
            None,
            f"<div class='status-pill status-danger'>❌ Chunking error: {str(e)}</div>",
            "_Indexing failed._",
            "No context available."
        )

    # Step 3: Fast FAISS vector indexing
    progress(0.75, desc="Building FAISS vector index...")
    try:
        emb_model = get_or_load_embedding_model()
        vector_store = create_vector_store(chunks, emb_model)
    except Exception as e:
        return (
            None,
            f"<div class='status-pill status-danger'>❌ Indexing error: {str(e)}</div>",
            "_Indexing failed._",
            "No context available."
        )

    progress(1.0, desc="Ready!")

    file_list_md = "\n".join([f"- 📄 **{name}**" for name in file_names])
    doc_stats_md = (
        f"**Indexed Files ({len(file_names)}):**\n"
        f"{file_list_md}\n\n"
        f"• **Chunks:** `{len(chunks)}` &nbsp;|&nbsp; **Vector Index:** `FAISS Ready ✅`"
    )

    success_msg = (
        f"<div class='status-pill status-success'>"
        f"✅ <b>Ready:</b> Indexed {len(file_names)} file(s) into {len(chunks)} chunks in sub-seconds."
        f"</div>"
    )

    ready_preview = (
        f"**Vector Store Active**\n\n"
        f"Indexed **{len(file_names)}** file(s) across **{len(chunks)}** chunks.\n"
        f"Ask any question in the chat to see real-time retrieved sources here."
    )

    return vector_store, success_msg, doc_stats_md, ready_preview


def chat_and_retrieve(
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
    vector_store: Any,
    api_key: str,
    model_name: str,
    top_k: int
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Processes user queries against the FAISS vector database and OpenRouter LLM.
    """
    if chat_history is None:
        chat_history = []
    else:
        chat_history = list(chat_history)
        
    user_query = (user_message or "").strip()
    if not user_query:
        return "", chat_history, "Please enter a question."

    if vector_store is None:
        bot_reply = (
            "⚠️ **No documents indexed yet.**\n\n"
            "Please upload your documents on the left panel and click **'⚡ Process & Index Documents'** first."
        )
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": bot_reply})
        return "", chat_history, "⚠️ No vector store active. Upload documents to begin."

    # Step 1: Retrieve relevant chunks from FAISS
    try:
        retrieved_results = retrieve_relevant_chunks(vector_store, user_query, top_k=top_k)
    except Exception as e:
        bot_reply = f"❌ **Retrieval error:** {str(e)}"
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": bot_reply})
        return "", chat_history, f"Retrieval Error: {str(e)}"

    if not retrieved_results:
        bot_reply = "I'm sorry, but based on the provided documents, I could not find information to answer this question."
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": bot_reply})
        return "", chat_history, "No matching document chunks found."

    # Step 2: Generate grounded answer via OpenRouter
    try:
        result = generate_answer_with_openrouter(
            query=user_query,
            retrieved_chunks=retrieved_results,
            api_key=api_key.strip() if api_key else None,
            model_name=model_name.strip() if model_name else None
        )
        answer = result["answer"]
        sources = result["sources"]
        used_model = result["model"]
    except Exception as e:
        bot_reply = f"❌ **Error generating response:** {str(e)}"
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": bot_reply})
        return "", chat_history, f"LLM Generation Error: {str(e)}"

    # Format Citation Badges
    citation_tags = [f"`{s['source']} (p. {s['page']})`" for s in sources]
    citations_line = " &nbsp;•&nbsp; ".join(citation_tags) if citation_tags else "Uploaded Documents"
    
    formatted_chat_reply = (
        f"{answer}\n\n"
        f"---\n"
        f"<small>📌 **Sources:** {citations_line} &nbsp;|&nbsp; 🤖 **Model:** `{used_model}`</small>"
    )

    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": formatted_chat_reply})

    # Format Context Inspector detailed view
    sources_md_blocks = [
        f"### 🔍 Retrieved Context for: *\"{user_query}\"*\n"
        f"**Model:** `{used_model}` &nbsp;|&nbsp; **Top-k Chunks:** {len(sources)}\n"
    ]

    for s in sources:
        score_info = f"(Distance Score: `{s['score']}`)" if s['score'] is not None else ""
        sources_md_blocks.append(
            f"#### 📄 Source {s['index']}: `{s['source']}` (Page {s['page']} | Chunk #{s['chunk_id']}) {score_info}\n"
            f"> {s['snippet'].replace(chr(10), chr(10) + '> ')}\n"
        )

    sources_full_md = "\n\n".join(sources_md_blocks)

    return "", chat_history, sources_full_md


def reset_application() -> Tuple[Any, None, str, List, str, str, str]:
    """Resets the entire RAG state."""
    return (
        None,
        None,
        "",
        [],
        "<div class='status-pill status-info'>ℹ️ Application reset. Ready for new documents.</div>",
        "_No documents currently indexed._",
        "Upload documents to inspect retrieved context snippets here."
    )


# ==============================================================================
# MODERN DARK THEME STYLING (DEEP CHARCOAL & PURPLE/BLUE ACCENTS)
# ==============================================================================
CUSTOM_CSS = """
/* Deep Charcoal Background & Modern Dark Theme */
:root, html, body, .gradio-container, gradio-app {
    background-color: #0f1117 !important;
    background: #0f1117 !important;
    color: #f8fafc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

body {
    background: #0f1117 !important;
    color: #f8fafc !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background-color: #0f1117 !important;
    color: #f8fafc !important;
}

/* Minimalist Clean Navbar */
.app-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: #1a1b26 !important;
    border: 1px solid #282a3c !important;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
}

.brand-title {
    font-size: 21px;
    font-weight: 700;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.4px;
}

.brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #818cf8;
    box-shadow: 0 0 12px #6366f1;
}

.nav-tag {
    font-size: 12.5px;
    font-weight: 600;
    color: #a5b4fc !important;
    background: #252738 !important;
    border: 1px solid #373a52;
    padding: 6px 14px;
    border-radius: 20px;
}

/* Typography Hierarchy: Crisp White Headings & Muted Secondary Text */
h1, h2, h3, h4, h5, h6, strong, b {
    color: #ffffff !important;
    font-weight: 700 !important;
}

p, span, label, .prose, .label-wrap {
    color: #cbd5e1 !important;
}

.muted-text, small, .info, .form-label-info {
    color: #94a3b8 !important;
}

/* Dark Gray Cards, Panels & Groups */
.gr-box, .gr-panel, .gr-group, .gr-form, .block, .accordion, fieldset, .wrap {
    background-color: #1a1b26 !important;
    border-color: #282a3c !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

/* Document Upload Dropzone */
.file-preview-holder, .upload-container, .dropzone, input[type="file"], .file-upload {
    background-color: #141520 !important;
    border: 1.5px dashed #3a3d54 !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease-in-out;
}

.file-preview-holder:hover, .upload-container:hover {
    border-color: #818cf8 !important;
    background-color: #181926 !important;
}

/* Text Inputs, Sliders & Dropdowns */
input, textarea, select, .gr-input, .gr-text-input {
    background-color: #141520 !important;
    color: #f8fafc !important;
    border: 1px solid #282a3c !important;
    border-radius: 10px !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    outline: none !important;
}

/* Chat Area Container */
#chat_window, .chatbot {
    background: #141520 !important;
    border: 1px solid #282a3c !important;
    border-radius: 14px !important;
}

/* Chat Messages: Purple Accent for User & Dark Gray for Bot */
.message.user, [data-testid="user"] {
    background: linear-gradient(135deg, #3730a3 0%, #4338ca 100%) !important;
    border: 1px solid #4f46e5 !important;
    color: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 15px rgba(67, 56, 202, 0.2);
}

.message.bot, [data-testid="bot"] {
    background: #1e1f2b !important;
    border: 1px solid #2d3043 !important;
    color: #f8fafc !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

/* Primary Action Buttons (Purple / Blue Gradient Glow) */
button.primary, #btn_process, #btn_send, .btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #3b82f6 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
}

button.primary:hover, #btn_process:hover, #btn_send:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
}

/* Secondary Action Buttons & Pills */
button.secondary, .btn-secondary {
    background: #232534 !important;
    color: #cbd5e1 !important;
    border: 1px solid #33364a !important;
    border-radius: 8px !important;
    transition: all 0.2s ease;
}

button.secondary:hover {
    background: #2c2f42 !important;
    color: #ffffff !important;
    border-color: #6366f1 !important;
}

/* Status Badges */
.status-pill {
    padding: 11px 16px;
    border-radius: 10px;
    font-size: 13.5px;
    margin-top: 8px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
}

.status-success {
    background-color: rgba(16, 185, 129, 0.12) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}

.status-warning {
    background-color: rgba(245, 158, 11, 0.12) !important;
    color: #fcd34d !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
}

.status-danger {
    background-color: rgba(239, 68, 68, 0.12) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}

.status-info {
    background-color: #1a1b26 !important;
    color: #94a3b8 !important;
    border: 1px solid #282a3c !important;
}

/* Context & Source Inspector */
.source-inspector {
    max-height: 480px;
    overflow-y: auto;
    padding: 14px;
    font-size: 13px;
    line-height: 1.6;
    background: #141520 !important;
    border-radius: 10px;
    border: 1px solid #282a3c;
    color: #cbd5e1 !important;
}
"""


# ==============================================================================
# GRADIO APPLICATION BUILDER
# ==============================================================================

def create_app() -> gr.Blocks:
    """Builds the clean, professional Gradio web interface."""
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")
    default_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    with gr.Blocks(title="Ask My Documents") as demo:
        
        vector_store_state = gr.State(value=None)

        # Professional Minimalist Header
        gr.HTML("""
        <div class="app-navbar">
            <div class="brand-title">
                <span class="brand-dot"></span>
                <span>Ask My Documents</span>
            </div>
            <div class="nav-tag">
                Fast RAG Engine &bull; OpenRouter AI
            </div>
        </div>
        """)

        with gr.Row():
            
            # ------------------------------------------------------------------
            # LEFT SIDEBAR: Document Management & Settings
            # ------------------------------------------------------------------
            with gr.Column(scale=4):
                
                with gr.Group():
                    file_input = gr.File(
                        label="Upload PDF, DOCX, or TXT Documents",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath",
                        elem_id="doc_uploader"
                    )
                    
                    process_btn = gr.Button(
                        "⚡ Process & Index Documents",
                        variant="primary",
                        size="lg",
                        elem_id="btn_process"
                    )

                # Real-time Status Banner
                status_output = gr.HTML(
                    value="<div class='status-pill status-info'>ℹ️ Upload documents to begin indexing.</div>",
                    elem_id="status_box"
                )

                # Document Index Stats
                with gr.Accordion("📊 Document Index Statistics", open=True):
                    doc_stats_output = gr.Markdown(
                        value="_No documents currently indexed._",
                        elem_id="doc_stats"
                    )

                # Model & Retrieval Settings
                with gr.Accordion("⚙️ Settings & Configuration", open=False):
                    api_key_input = gr.Textbox(
                        label="OpenRouter API Key",
                        placeholder="sk-or-v1-...",
                        value=default_api_key,
                        type="password",
                        info="Configured securely via .env file or editable here."
                    )
                    
                    model_dropdown = gr.Dropdown(
                        label="LLM Model",
                        choices=[
                            "openai/gpt-4o-mini",
                            "google/gemini-2.0-flash-001",
                            "meta-llama/llama-3.3-70b-instruct:free",
                            "anthropic/claude-3.5-haiku",
                            "mistralai/mistral-7b-instruct",
                            "deepseek/deepseek-chat",
                        ],
                        value=default_model,
                        allow_custom_value=True,
                        info="Select a model or enter any custom OpenRouter model slug."
                    )

                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Top-K Retrieved Chunks"
                    )

                    chunk_size_slider = gr.Slider(
                        minimum=200,
                        maximum=2000,
                        value=1000,
                        step=100,
                        label="Chunk Size (Characters)"
                    )

                    chunk_overlap_slider = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=150,
                        step=50,
                        label="Chunk Overlap (Characters)"
                    )

                reset_btn = gr.Button("🗑️ Clear & Reset All", variant="secondary", size="sm")

            # ------------------------------------------------------------------
            # RIGHT COLUMN: Chat Area & Context Inspector
            # ------------------------------------------------------------------
            with gr.Column(scale=6):
                
                chatbot = gr.Chatbot(
                    label="💬 Q&A Assistant",
                    height=460,
                    elem_id="chat_window"
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about your documents...",
                        lines=2,
                        scale=8,
                        elem_id="query_box"
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=2, elem_id="btn_send")

                with gr.Row():
                    clear_chat_btn = gr.Button("🧹 Clear Chat", size="sm", scale=3)
                    sample_q1 = gr.Button("💡 Summarize key points", size="sm", scale=3)
                    sample_q2 = gr.Button("💡 What are the main takeaways?", size="sm", scale=4)

                with gr.Accordion("🔍 Retrieved Sources & Context Inspector", open=True):
                    sources_inspector = gr.Markdown(
                        value="Upload documents to inspect retrieved context snippets here.",
                        elem_classes=["source-inspector"]
                    )

        # ----------------------------------------------------------------------
        # EVENT BINDINGS
        # ----------------------------------------------------------------------
        process_btn.click(
            fn=process_uploaded_files,
            inputs=[file_input, chunk_size_slider, chunk_overlap_slider],
            outputs=[vector_store_state, status_output, doc_stats_output, sources_inspector]
        )

        send_inputs = [
            query_input,
            chatbot,
            vector_store_state,
            api_key_input,
            model_dropdown,
            top_k_slider
        ]
        send_outputs = [query_input, chatbot, sources_inspector]

        send_btn.click(fn=chat_and_retrieve, inputs=send_inputs, outputs=send_outputs)
        query_input.submit(fn=chat_and_retrieve, inputs=send_inputs, outputs=send_outputs)

        sample_q1.click(
            fn=lambda: "Summarize the key points of the uploaded documents.",
            inputs=None,
            outputs=[query_input]
        )
        sample_q2.click(
            fn=lambda: "What are the main takeaways or conclusions?",
            inputs=None,
            outputs=[query_input]
        )

        clear_chat_btn.click(
            fn=lambda: ("", [], "Chat cleared. Ready for questions."),
            inputs=None,
            outputs=[query_input, chatbot, sources_inspector]
        )

        reset_btn.click(
            fn=reset_application,
            inputs=None,
            outputs=[
                vector_store_state,
                file_input,
                query_input,
                chatbot,
                status_output,
                doc_stats_output,
                sources_inspector
            ]
        )

    return demo


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = create_app()
    
    # Read host and port:
    # Render and cloud platforms require binding to 0.0.0.0 and listening on $PORT (default 10000)
    # Local Windows machine uses 127.0.0.1:7860 so the URL is directly clickable in your browser
    port_env = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    if port_env:
        server_port = int(port_env)
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    else:
        server_port = 7860 if os.name == "nt" else 10000
        server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1" if os.name == "nt" else "0.0.0.0")

    # Dark Theme Token Definitions (Deep Charcoal & Dark Gray)
    dark_theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0f1117",
        body_background_fill_dark="#0f1117",
        body_text_color="#f8fafc",
        body_text_color_dark="#f8fafc",
        block_background_fill="#1a1b26",
        block_background_fill_dark="#1a1b26",
        block_border_color="#282a3c",
        block_border_color_dark="#282a3c",
        panel_background_fill="#1a1b26",
        panel_background_fill_dark="#1a1b26",
        background_fill_primary="#0f1117",
        background_fill_primary_dark="#0f1117",
        background_fill_secondary="#1a1b26",
        background_fill_secondary_dark="#1a1b26",
    )

    head_scripts = """
    <script>
        document.documentElement.classList.add('dark');
        document.body.classList.add('dark');
    </script>
    <style>
        html, body, .gradio-container, gradio-app {
            background-color: #0f1117 !important;
            background: #0f1117 !important;
            color: #f8fafc !important;
        }
    </style>
    """

    print(f"[INFO] Launching Ask My Documents RAG on {server_name}:{server_port}...")

    app.launch(
        server_name=server_name,
        server_port=server_port,
        theme=dark_theme,
        css=CUSTOM_CSS,
        head=head_scripts,
        js="() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }",
        show_error=True,
        share=False
    )


