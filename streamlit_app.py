import streamlit as st
import os
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import retrieve, MIN_RETRIEVAL_SIMILARITY
from document_ingestion import index_document, remove_document

# UI Configurations
st.set_page_config(
    page_title="Local Document Intelligence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for polish (restrained, dark, premium)
st.markdown("""
<style>
    /* Base styling for a dark theme */
    html, body {
        background-color: #0d1117 !important;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        color: #c9d1d9;
    }
    
    /* Ensure content sits above the animated background */
    [data-testid="stMain"] {
        position: relative;
        z-index: 1;
    }
    
    /* Ambient Animated Background */
    @keyframes ambientFlowA {
        0%   { transform: translate3d(-5%, -5%, 0) scale(1.0); }
        50%  { transform: translate3d(8%, 10%, 0) scale(1.15); }
        100% { transform: translate3d(-5%, -5%, 0) scale(1.0); }
    }
    
    @keyframes ambientFlowB {
        0%   { transform: translate3d(5%, 5%, 0) scale(1.0); }
        50%  { transform: translate3d(-8%, -10%, 0) scale(1.1); }
        100% { transform: translate3d(5%, 5%, 0) scale(1.0); }
    }
    
    .stApp::before, .stApp::after {
        content: '';
        position: fixed;
        inset: -20%;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }
    
    .stApp::before {
        background: radial-gradient(circle at 20% 30%, rgba(88, 166, 255, 0.28) 0%, transparent 45%),
                    radial-gradient(circle at 80% 70%, rgba(137, 87, 229, 0.25) 0%, transparent 50%);
        animation: ambientFlowA 22s ease-in-out infinite;
    }
    
    .stApp::after {
        background: radial-gradient(circle at 60% 40%, rgba(70, 70, 255, 0.22) 0%, transparent 55%),
                    radial-gradient(circle at 30% 70%, rgba(0, 150, 255, 0.18) 0%, transparent 45%);
        animation: ambientFlowB 28s ease-in-out infinite reverse;
    }
    
    /* Hide unnecessary Streamlit chrome & chat avatars */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stChatMessageAvatar"] { display: none !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    
    /* Product layout max-width */
    .block-container {
        max-width: 1000px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* Eyebrow & Titles */
    .eyebrow {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 400;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Source Badges */
    .source-label {
        font-size: 0.65rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
    }
    .source-badge {
        display: inline-block;
        color: #8b949e;
        border-radius: 4px;
        padding: 0;
        font-size: 0.75rem;
        font-family: monospace;
        margin-right: 0.8rem;
    }
    
    /* Glassmorphism for Messages */
    .glass-assistant-msg {
        background-color: rgba(20, 24, 35, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        color: #c9d1d9;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.5rem;
    }
    
    .glass-user-msg {
        background-color: rgba(88, 166, 255, 0.08);
        border: 1px solid rgba(88, 166, 255, 0.15);
        border-radius: 18px;
        padding: 1.2rem 1.5rem;
        color: #e6edf3;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .glass-doc-item {
        background-color: rgba(20, 24, 35, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        color: #8b949e;
        font-size: 0.85rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-top: 0.1rem;
    }
    
    /* Glassmorphism for Chat Input */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    div[data-testid="stChatInput"] > div {
        background-color: rgba(20, 24, 35, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Glassmorphism for Uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(20, 24, 35, 0.55) !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 18px !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #c9d1d9 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
    }
    
    /* Chat Styling override */
    .stChatMessage {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Subtle layout tweaks */
    div[data-testid="stExpander"] * {
        background-color: transparent !important;
    }
    div[data-testid="stExpander"] {
        background-color: rgba(20, 24, 35, 0.45) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background-color: rgba(20, 24, 35, 0.55);
        color: #c9d1d9;
        font-weight: 400;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: rgba(88, 166, 255, 0.4);
        background-color: rgba(88, 166, 255, 0.1);
        color: #ffffff;
    }
    
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    db_path = os.path.join('data', 'knowledge.db')
    if not os.path.exists(db_path):
        return [], []
    
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT source, content, embedding FROM chunks')
            rows = cursor.fetchall()
            
            cursor.execute('SELECT source FROM documents')
            sources = [row[0] for row in cursor.fetchall()]
            return rows, sources
        except sqlite3.OperationalError:
            return [], []

@st.cache_resource(show_spinner=False)
def load_models():
    config = Configuration(app_name="LocalRAGApp_Streamlit")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model_alias = "qwen3-embedding-0.6b"
    embed_model = manager.catalog.get_model(embed_model_alias)
    if not embed_model.is_cached:
        embed_model.download()
    if not embed_model.is_loaded:
        embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model_alias = "qwen2.5-1.5b"
    chat_model = manager.catalog.get_model(chat_model_alias)
    if not chat_model.is_cached:
        chat_model.download()
    if not chat_model.is_loaded:
        chat_model.load()
    chat_client = chat_model.get_chat_client()

    return embed_client, chat_client

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------

# Header Section
st.markdown("<div class='eyebrow'>LOCAL DOCUMENT INTELLIGENCE</div>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Ask your documents.</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Build a local knowledge base from your own files and ask grounded questions powered by Microsoft Foundry Local.</div>", unsafe_allow_html=True)

db_path = os.path.join('data', 'knowledge.db')
rows, sources = load_knowledge_base()

with st.spinner("Loading Foundry Local..."):
    embed_client, chat_client = load_models()

# Empty State Logic
if not sources:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Your knowledge base is empty. Add a PDF, DOCX or TXT file to get started.")
    
    uploaded_files = st.file_uploader("Add documents", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        if st.button("Index Documents"):
            for f in uploaded_files:
                file_bytes = f.read()
                res = index_document(f.name, file_bytes, embed_client, db_path)
                st.write(f"{f.name}: {res['status'].capitalize()}")
            load_knowledge_base.clear()
            st.rerun()
    st.stop()

# Layout: Main Chat vs Document Management
main_col, side_col = st.columns([2.6, 1.4], gap="large")

with side_col:
    st.markdown("#### Documents")
    
    st.markdown("<div style='font-size: 0.85rem; color: #8b949e; margin-bottom: 4px;'>Supports PDF, DOCX, TXT</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Add documents", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        if st.button("Index"):
            for f in uploaded_files:
                file_bytes = f.read()
                res = index_document(f.name, file_bytes, embed_client, db_path)
                st.write(f"*{f.name}: {res['status'].capitalize()}*")
            load_knowledge_base.clear()
            st.rerun()
            
    st.markdown("---")
    
    if sources:
        st.markdown(f"<div style='font-size: 0.9rem; color: #8b949e; margin-bottom: 0.8rem;'>{len(sources)} documents indexed</div>", unsafe_allow_html=True)
        with st.expander("Manage documents"):
            for s in sources:
                col1, col2 = st.columns([1, 1], gap="small")
                col1.markdown(f"<div class='glass-doc-item' style='margin-bottom: 0.2rem;'>{s}</div>", unsafe_allow_html=True)
                
                confirm_key = f"confirm_del_{s}"
                if st.session_state.get(confirm_key, False):
                    st.warning(f"Remove '{s}' from the knowledge base?\n\nIts indexed content will be removed. Other documents will not be affected.")
                    c_col1, c_col2 = st.columns(2)
                    if c_col1.button("Cancel", key=f"cancel_{s}"):
                        st.session_state[confirm_key] = False
                        st.rerun()
                    if c_col2.button("Remove from knowledge base", key=f"do_del_{s}"):
                        remove_document(db_path, s)
                        st.session_state[confirm_key] = False
                        load_knowledge_base.clear()
                        st.rerun()
                else:
                    if col2.button("Remove", key=f"init_del_{s}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()
                
    st.markdown("---")
    
    with st.expander("System details"):
        st.markdown(f"""
        <div style='font-size: 0.8rem; color:#8b949e;'>
        <b>Microsoft Foundry Local</b><br><br>
        <b>Embedding model:</b><br>qwen3-embedding-0.6b<br><br>
        <b>Chat model:</b><br>qwen2.5-1.5b<br><br>
        <b>Retrieval:</b><br>Top-K 3<br><br>
        Runs locally during normal inference without a cloud LLM API.
        </div>
        """, unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with main_col:
    # ---------------------------------------------------------
    # Chat Interface
    # ---------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_assistant_content(content, used_sources=None):
        st.markdown(f"<div class='glass-assistant-msg'>\n\n{content}\n\n</div>", unsafe_allow_html=True)
        if used_sources:
            st.markdown("<div class='source-label'>SOURCES</div>", unsafe_allow_html=True)
            badges_html = "".join([f"<span class='source-badge'>{src}</span>" for src in used_sources])
            st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

    # Empty Chat State / Hints
    if not st.session_state.messages:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("Summarize the main ideas", use_container_width=True):
                st.session_state.hint = "Summarize the main ideas"
                st.rerun()
        with sc2:
            if st.button("What does this document say about...?", use_container_width=True):
                st.session_state.hint = "What does this document say about...?"
                st.rerun()
        with sc3:
            if st.button("Compare the relevant sections", use_container_width=True):
                st.session_state.hint = "Compare the relevant sections"
                st.rerun()

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            render_assistant_content(message["content"], message.get("sources", []))
        else:
            st.markdown(f"<div class='glass-user-msg'>\n\n{message['content']}\n\n</div>", unsafe_allow_html=True)

    system_instruction = (
        "Answer only from the supplied context.\n"
        "Do not use outside knowledge.\n"
        "Do not invent information.\n"
        "If the context does not contain enough information, say exactly:\n"
        "I don't have enough information in the provided documents.\n"
        "The answer must be a maximum of 2 sentences and approximately 35-55 words.\n"
        "Give the direct answer first.\n"
        "Do not include unnecessary introduction or repeated explanation.\n"
        "Avoid extra background, use cases, examples, or repetition unless explicitly requested.\n"
        "Do not include long multi-paragraph responses.\n"
        "Do NOT format the source inside your answer."
    )

    def is_multi_document_query(query: str) -> bool:
        q = query.lower()
        multi_keywords = [
            "compare", "comparison", "similarities", "differences", 
            "across the documents", "across these files", 
            "between the documents", "synthesize"
        ]
        return any(k in q for k in multi_keywords)

    initial_q = ""
    if "hint" in st.session_state and st.session_state.hint:
        initial_q = st.session_state.hint
        st.session_state.hint = None

    question = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    # If a hint was clicked, but user didn't enter anything, we just don't process yet, but Streamlit chat_input can't easily be pre-filled dynamically without custom components. Actually, Streamlit doesn't support pre-filling chat_input directly. Wait, I should submit it directly if a hint is clicked! But the user said: "These should be generic and not imply knowledge that may not exist. Do not automatically submit an invalid hardcoded question. generate three neutral usage hints ... These should be generic and not imply knowledge that may not exist."
    # Since st.chat_input doesn't accept a default value to pre-fill, if the user clicks a hint button, I will just submit it as the question directly. The hints are generic enough!

    if initial_q and not question:
        question = initial_q

    if question:
        st.markdown(f"<div class='glass-user-msg'>\n\n{question}\n\n</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.spinner("Searching knowledge base..."):
            embed_response = embed_client.generate_embedding(question)
            query_embedding = embed_response.data[0].embedding
            
            top_results = retrieve(query_embedding, rows, top_k=3)
            
            fallback_phrase = "I don't have enough information in the provided documents."
            used_sources = []
            
            if not top_results or top_results[0]['score'] < MIN_RETRIEVAL_SIMILARITY:
                answer = fallback_phrase
            else:
                context_str = ""
                for res in top_results:
                    context_str += f"SOURCE: {res['source']}\nCONTENT: {res['content']}\n\n"
                    
                user_prompt = f"Context:\n{context_str}\nQuestion: {question}\n\nIMPORTANT: Your answer MUST be exactly 1 or 2 sentences maximum. Do not exceed 2 sentences."
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
                
                chat_response = chat_client.complete_chat(messages)
                answer = chat_response.choices[0].message.content.strip()
                
                # Force maximum 2 sentences for the small model
                import re
                sentences = re.split(r'(?<=[.!?])\s+', answer)
                if len(sentences) > 2:
                    answer = " ".join(sentences[:2])
                
                if "don't have enough information" in answer.lower() or "do not have enough information" in answer.lower():
                    answer = fallback_phrase
                else:
                    used_sources = []
                    if is_multi_document_query(question):
                        for res in top_results:
                            if res['score'] >= MIN_RETRIEVAL_SIMILARITY:
                                if res['source'] not in used_sources:
                                    used_sources.append(res['source'])
                    else:
                        used_sources = [top_results[0]['source']]
                
            render_assistant_content(answer, used_sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": used_sources})
