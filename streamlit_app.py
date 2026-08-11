import streamlit as st
import os
import json
import sqlite3
import math
from foundry_local_sdk import Configuration, FoundryLocalManager

# UI Configurations
st.set_page_config(
    page_title="Local RAG AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for polish (restrained)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions (Exact RAG flow)
# ---------------------------------------------------------

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

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
            
            # Get unique sources for the sidebar
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

    # Embedding Model
    embed_model_alias = "qwen3-embedding-0.6b"
    embed_model = manager.catalog.get_model(embed_model_alias)
    if not embed_model.is_cached:
        embed_model.download()
    if not embed_model.is_loaded:
        embed_model.load()
    embed_client = embed_model.get_embedding_client()

    # Chat Model
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

st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Local RAG AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 1.1em; margin-bottom: 5px;'>Local document intelligence powered by Microsoft Foundry Local</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Ask grounded questions about your local knowledge base.</p>", unsafe_allow_html=True)

# Check database
rows, sources = load_knowledge_base()
if rows is None:
    st.error("Knowledge base not found. Run: `python ingest.py`")
    st.stop()

# Status row cards
c1, c2, c3 = st.columns(3)
c1.markdown("<div style='text-align: center; padding: 10px; background-color: rgba(255,255,255,0.05); border-radius: 8px;'><b>LOCAL</b></div>", unsafe_allow_html=True)
c2.markdown(f"<div style='text-align: center; padding: 10px; background-color: rgba(255,255,255,0.05); border-radius: 8px;'><b>{len(rows)} CHUNKS</b></div>", unsafe_allow_html=True)
c3.markdown(f"<div style='text-align: center; padding: 10px; background-color: rgba(255,255,255,0.05); border-radius: 8px;'><b>{len(sources)} SOURCES</b></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Check models
with st.spinner("Loading models (this might take a moment)..."):
    embed_client, chat_client = load_models()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("Microsoft Foundry Local")
    
    st.subheader("Models")
    st.markdown("- Embedding: `qwen3-embedding-0.6b`\n- Chat: `qwen2.5-1.5b`")
    
    st.subheader("Retrieval")
    st.markdown("- Top-K: 3")
    
    st.subheader("Knowledge Base")
    st.markdown(f"{len(sources)} sources\n\n{len(rows)} chunks")
    with st.expander("View documents"):
        if sources:
            for s in sources:
                st.markdown(f"- `{s}`")

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
def render_assistant_content(content, sources=None):
    st.markdown(content)
    if sources:
        sources_str = ", ".join(sources)
        st.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px 16px; border-radius: 8px; margin-top: 16px; border: 1px solid rgba(255, 255, 255, 0.08);">
            <div style="font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; color: #888; margin-bottom: 4px;">Source</div>
            <div style="font-size: 0.9em; font-family: monospace; color: #eee;">{sources_str}</div>
        </div>
        """, unsafe_allow_html=True)

if not st.session_state.messages:
    st.chat_message("assistant").markdown("Ask a question about the documents in the local knowledge base.")
    
    st.markdown("<div style='margin-bottom: 10px;'><small><b>Try an example:</b></small></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is RAG?", use_container_width=True):
            st.session_state.example_q = "What is RAG?"
    with col2:
        if st.button("What is Python?", use_container_width=True):
            st.session_state.example_q = "What is Python?"
    with col3:
        if st.button("Why SQLite?", use_container_width=True):
            st.session_state.example_q = "Why does SQLite not need a separate server?"

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
        with st.spinner("Searching local knowledge base..."):
            # 1. Generate Query Embedding
            embed_response = embed_client.generate_embedding(question)
            query_embedding = embed_response.data[0].embedding
            
            # 2. Similarity Search
            results = []
            for row in rows:
                source, content, embedding_json = row
                doc_embedding = json.loads(embedding_json)
                sim = cosine_similarity(query_embedding, doc_embedding)
                results.append({
                    "source": source,
                    "content": content,
                    "score": sim
                })
                
            results.sort(key=lambda x: x["score"], reverse=True)
            top_k = 3
            top_results = results[:top_k]
            
            # 3. Context Construction
            context_str = ""
            for res in top_results:
                context_str += f"SOURCE: {res['source']}\nCONTENT: {res['content']}\n\n"
                
            user_prompt = f"Context:\n{context_str}\nQuestion: {question}"
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            
            # 4. Generate Answer
            chat_response = chat_client.complete_chat(messages)
            answer = chat_response.choices[0].message.content.strip()
            
            fallback_phrase = "I don't have enough information in the provided documents."
            if "don't have enough information" in answer.lower() or "do not have enough information" in answer.lower():
                answer = fallback_phrase
                used_sources = []
            else:
                used_sources = list(set([res['source'] for res in top_results]))
                
            render_assistant_content(answer, used_sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": used_sources})
