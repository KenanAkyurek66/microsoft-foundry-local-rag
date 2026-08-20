import streamlit as st
import os
import json
import sqlite3
import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import retrieve, MIN_RETRIEVAL_SIMILARITY

# UI Configurations
st.set_page_config(
    page_title="Local RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for polish (restrained, dark, premium)
st.markdown("""
<style>
    /* Base styling for a dark theme */
    .stApp {
        background-color: #0d1117;
    }
    
    /* Hide unnecessary Streamlit chrome but keep header for expand/collapse sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Product layout max-width */
    .block-container {
        max-width: 1000px !important;
        padding-top: 2rem !important;
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
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Status Cards */
    .status-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .status-card {
        flex: 1;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
    }
    .status-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* System Panel */
    .system-panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
    }
    .sys-group {
        margin-bottom: 1.2rem;
    }
    .sys-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.2rem;
    }
    .sys-value {
        font-size: 0.9rem;
        color: #e6edf3;
        font-family: monospace;
    }

    /* Source Badges */
    .source-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        margin-top: 1rem;
    }
    .source-badge {
        display: inline-block;
        background-color: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.2);
        color: #58a6ff;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-family: monospace;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Example Chips */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #30363d;
        background-color: rgba(255,255,255,0.05);
        color: #c9d1d9;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #58a6ff;
        color: #fff;
        background-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions (Exact RAG flow)
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    db_path = os.path.join('data', 'knowledge.db')
    if not os.path.exists(db_path):
        return None, None
    
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT source, content, embedding FROM chunks')
            rows = cursor.fetchall()
            if not rows:
                return None, None
            
            cursor.execute('SELECT DISTINCT source FROM chunks')
            sources = [row[0] for row in cursor.fetchall()]
            return rows, sources
        except sqlite3.OperationalError:
            return None, None

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
st.markdown("<div class='eyebrow'>LOCAL KNOWLEDGE ASSISTANT</div>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Local RAG AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Grounded answers from your local documents, powered by Microsoft Foundry Local.</div>", unsafe_allow_html=True)

# Load Data
rows, sources = load_knowledge_base()
if rows is None:
    st.error("Knowledge base not found. Run: `python ingest.py`")
    st.stop()

num_chunks = len(rows)
num_sources = len(sources)

# Three compact premium status cards
st.markdown(f"""
<div class='status-container'>
    <div class='status-card'>
        <div class='status-value'>LOCAL</div>
        <div class='status-label'>Inference</div>
    </div>
    <div class='status-card'>
        <div class='status-value'>{num_chunks}</div>
        <div class='status-label'>Chunks</div>
    </div>
    <div class='status-card'>
        <div class='status-value'>{num_sources}</div>
        <div class='status-label'>Sources</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Initializing models..."):
    embed_client, chat_client = load_models()

# Layout: Main Chat vs System Panel
main_col, sys_col = st.columns([2.5, 1], gap="large")

with sys_col:
    # System Information Panel
    st.markdown(f"""
    <div class='system-panel'>
        <div class='sys-group'>
            <div class='sys-label'>System</div>
            <div class='sys-value'>Microsoft Foundry Local</div>
        </div>
        <div class='sys-group'>
            <div class='sys-label'>Models</div>
            <div class='sys-value' style='margin-bottom:4px;'>Embedding<br>qwen3-embedding-0.6b</div>
            <div class='sys-value'>Chat<br>qwen2.5-1.5b</div>
        </div>
        <div class='sys-group'>
            <div class='sys-label'>Retrieval</div>
            <div class='sys-value'>Top-K 3</div>
        </div>
        <div class='sys-group'>
            <div class='sys-label'>Knowledge Base</div>
            <div class='sys-value'>{num_sources} sources<br>{num_chunks} chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View indexed documents"):
        for s in sources:
            st.markdown(f"- `{s}`")
            
    # Utility Action
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with main_col:
    # ---------------------------------------------------------
    # Chat Interface
    # ---------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def render_assistant_content(content, used_sources=None):
        st.markdown(content)
        if used_sources:
            st.markdown("<div class='source-label'>SOURCE</div>", unsafe_allow_html=True)
            badges_html = "".join([f"<span class='source-badge'>{src}</span>" for src in used_sources])
            st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

    # Empty State Welcome
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Ask a grounded question about the documents in your knowledge base.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if st.button("What is RAG?", use_container_width=True):
                    st.session_state.example_q = "What is RAG?"
                    st.rerun()
            with sc2:
                if st.button("What is Python?", use_container_width=True):
                    st.session_state.example_q = "What is Python?"
                    st.rerun()
            with sc3:
                if st.button("Why SQLite?", use_container_width=True):
                    st.session_state.example_q = "Why does SQLite not need a separate server?"
                    st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_content(message["content"], message.get("sources", []))
            else:
                st.markdown(message["content"])

    # System Instruction
    system_instruction = (
        "Answer only from the supplied context.\n"
        "Do not use outside knowledge.\n"
        "Do not invent information.\n"
        "If the context does not contain enough information, say exactly:\n"
        "I don't have enough information in the provided documents.\n"
        "The answer must be a maximum of 2 to 3 sentences and maximum 70 words.\n"
        "Directly answer the question with no unnecessary introduction and no repeated explanation.\n"
        "Avoid extra background, use cases, examples, or repetition unless explicitly requested.\n"
        "Do not include long multi-paragraph responses.\n"
        "Do NOT format the source inside your answer."
    )

    # Chat input
    question = st.chat_input("Ask a question...")
    
    # Handle example chip clicks
    if "example_q" in st.session_state and st.session_state.example_q:
        question = st.session_state.example_q
        st.session_state.example_q = None

    if question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # 1. Generate Query Embedding
                embed_response = embed_client.generate_embedding(question)
                query_embedding = embed_response.data[0].embedding
                
                # 2. Semantic Retrieval
                top_results = retrieve(query_embedding, rows, top_k=3)
                
                fallback_phrase = "I don't have enough information in the provided documents."
                used_sources = []
                
                # 3. Confidence Gate
                if not top_results or top_results[0]['score'] < MIN_RETRIEVAL_SIMILARITY:
                    answer = fallback_phrase
                else:
                    # 4. Context Construction
                    context_str = ""
                    for res in top_results:
                        context_str += f"SOURCE: {res['source']}\nCONTENT: {res['content']}\n\n"
                        
                    user_prompt = f"Context:\n{context_str}\nQuestion: {question}"
                    messages = [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # 5. Generate Answer
                    chat_response = chat_client.complete_chat(messages)
                    answer = chat_response.choices[0].message.content.strip()
                    
                    if "don't have enough information" in answer.lower() or "do not have enough information" in answer.lower():
                        answer = fallback_phrase
                    else:
                        used_sources = list(set([res['source'] for res in top_results]))
                    
                render_assistant_content(answer, used_sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": used_sources})
