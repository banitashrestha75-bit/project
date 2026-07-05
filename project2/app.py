import streamlit as st
import os
import uuid
import dotenv
from groq import Groq

# Load environment variables (finds .env walking up directory tree from script and CWD)
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=False))
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True), override=False)

# Map LLAMA_PARSE_API_KEY to LLAMA_CLOUD_API_KEY if needed
if "LLAMA_PARSE_API_KEY" in os.environ and "LLAMA_CLOUD_API_KEY" not in os.environ:
    os.environ["LLAMA_CLOUD_API_KEY"] = os.environ["LLAMA_PARSE_API_KEY"]

# Import project modules
import database
import auth
import parser
import rag_engine
import search_engine
import guardrails

# Initialize SQLite database
database.init_db()

# --- Page Configurations ---
st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Aesthetics (CSS) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0f0f12;
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #16161a !important;
        border-right: 1px solid #232329;
    }
    
    /* Titles and Headers */
    h1, h2, h3, .stHeader {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(22, 22, 26, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .login-container {
        max-width: 500px;
        margin: 5rem auto;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Chat Bubbles Customization */
    .chat-bubble-source {
        font-size: 0.8rem;
        color: #818cf8;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f0f12;
    }
    ::-webkit-scrollbar-thumb {
        background: #232329;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper to initialize Groq client ---
def get_groq_client():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        try:
            if "GROQ_API_KEY" in st.secrets:
                groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not groq_key:
        return None
    return Groq(api_key=groq_key)


# --- Session State Setup ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "temp_chunks" not in st.session_state:
    st.session_state.temp_chunks = []
if "temp_doc_name" not in st.session_state:
    st.session_state.temp_doc_name = None

# --- User Auth Screens (Login / Register) ---
if not st.session_state.logged_in:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>🤖 RAG Assistant Portal</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔒 Sign In", "📝 Create Account"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Log In", use_container_width=True)
            
            if submit:
                success, msg, user_data = auth.authenticate_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = user_data
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username").strip()
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_signup:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, msg = auth.register_user(new_username, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# --- LOGGED IN AREA ---
user_id = st.session_state.user["id"]
username = st.session_state.user["username"]

# Groq check
groq_client = get_groq_client()
if not groq_client:
    st.sidebar.error("⚠️ GROQ_API_KEY is not configured! Chat functionality will be disabled.")


# --- SIDEBAR: User Session, History & Settings ---
with st.sidebar:
    st.markdown(f"### 👋 Welcome, **{username.capitalize()}**")
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.current_chat_id = None
        st.session_state.temp_chunks = []
        st.session_state.temp_doc_name = None
        st.rerun()
        
    st.markdown("---")
    
    # Model Selection
    selected_model = st.selectbox(
        "🧠 LLM Model",
        options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        database.create_chat(chat_id, user_id, "New Chat Session")
        st.session_state.current_chat_id = chat_id
        st.rerun()
        
    chats = database.get_user_chats(user_id)
    
    if not chats:
        st.info("No active chat sessions. Start one above!")
    else:
        # If no current chat is selected, default to the most recent one
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = chats[0]["id"]
            
        for chat in chats:
            col1, col2 = st.columns([0.85, 0.15])
            
            # Selection button
            is_active = chat["id"] == st.session_state.current_chat_id
            style = "font-weight: bold; color: #a78bfa;" if is_active else ""
            
            with col1:
                if st.button(chat["title"], key=f"sel_{chat['id']}", use_container_width=True, help=chat["title"]):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}", help="Delete Chat"):
                    database.delete_chat(chat["id"])
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                    st.rerun()
                    
    st.markdown("---")
    st.markdown("### 📁 Indexed Documents")
    user_docs = database.get_user_uploaded_documents(user_id)
    if not user_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in user_docs:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.caption(f"📄 {doc}")
            with col2:
                if st.button("❌", key=f"deldoc_{doc}", help="Remove from database"):
                    database.delete_user_chunks(user_id, doc)
                    st.rerun()


# --- MAIN SCREEN ---
st.markdown('<div class="main-title">RAG Chatbot Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Chat with your documents or Fallback to Web Search via Tavily</div>', unsafe_allow_html=True)

# 1. DOCUMENT HUB EXPANDER
with st.expander("📂 Document Hub - Upload & Chunk Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload text documents (PDF, TXT, MD, JSON)",
        type=["pdf", "txt", "md", "json"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.write("---")
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            
            # Check if document is already indexed
            if file_name in user_docs:
                st.info(f"'{file_name}' is already indexed in your database. You can search against it.")
                continue
                
            col_parse, col_status = st.columns([0.2, 0.8])
            with col_parse:
                if st.button(f"⚙️ Parse: {file_name}", key=f"parse_{file_name}"):
                    # Save temporarily to parse
                    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, file_name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    with st.spinner(f"Parsing '{file_name}'..."):
                        try:
                            ext = os.path.splitext(file_name)[1]
                            parsed_text, parser_used = parser.parse_document(temp_path, ext)
                            
                            # Perform chunking
                            chunks = rag_engine.chunk_text(parsed_text)
                            
                            # Cache in session state for review
                            st.session_state.temp_chunks = chunks
                            st.session_state.temp_doc_name = file_name
                            
                            st.success(f"Successfully parsed '{file_name}' using {parser_used} into {len(chunks)} chunks!")
                        except Exception as e:
                            st.error(f"Error parsing file: {e}")
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                                
        # Show chunked data review if available
        if st.session_state.temp_chunks and st.session_state.temp_doc_name:
            st.markdown(f"#### 🔎 Preview Chunked Data: `{st.session_state.temp_doc_name}`")
            
            # Display chunks in an interactive table
            import pandas as pd
            df_chunks = pd.DataFrame({
                "Chunk Index": list(range(len(st.session_state.temp_chunks))),
                "Content Preview": [c[:120] + "..." if len(c) > 120 else c for c in st.session_state.temp_chunks]
            })
            st.dataframe(df_chunks, use_container_width=True)
            
            # Button to save chunks
            if st.button("💾 Save Chunks to Database", type="primary"):
                with st.spinner("Saving to database..."):
                    database.add_chunks(
                        user_id=user_id,
                        document_name=st.session_state.temp_doc_name,
                        chunks_list=st.session_state.temp_chunks
                    )
                    st.success(f"Indexed {len(st.session_state.temp_chunks)} chunks for '{st.session_state.temp_doc_name}'!")
                    st.session_state.temp_chunks = []
                    st.session_state.temp_doc_name = None
                    st.rerun()


# 2. CHAT DIALOGUE WINDOW
if st.session_state.current_chat_id is None:
    # If no session, create one
    chat_id = str(uuid.uuid4())
    database.create_chat(chat_id, user_id, "New Chat Session")
    st.session_state.current_chat_id = chat_id
    st.rerun()

current_chat_id = st.session_state.current_chat_id

# Display past messages
messages = database.get_chat_messages(current_chat_id)
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Chat Input
user_query = st.chat_input("Ask a question about your documents or any general topic...")

if user_query:
    user_query = user_query.strip()
    
    # 1. Render User Message immediately
    with st.chat_message("user"):
        st.write(user_query)
        
    database.add_message(current_chat_id, "user", user_query)
    
    # Rename chat if it is default
    chat_title = next((c["title"] for c in chats if c["id"] == current_chat_id), "New Chat Session")
    if chat_title == "New Chat Session":
        new_title = user_query[:35] + ("..." if len(user_query) > 35 else "")
        # Update db title
        conn = database.get_db_connection()
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, current_chat_id))
        conn.commit()
        conn.close()
        # Trigger history refresh on next run
        
    # 2. Apply Guardrails Pre-processing
    guard_triggered, guard_response = guardrails.handle_guardrails(user_query)
    
    if guard_triggered:
        with st.chat_message("assistant"):
            st.info(guard_response)
        database.add_message(current_chat_id, "assistant", guard_response)
        st.rerun()
        
    # 3. Perform Retrieval-Augmented Generation (RAG)
    if not groq_client:
        st.error("GROQ_API_KEY is not set. Cannot run LLM.")
        st.stop()
        
    # Retrieve local documents
    relevant_chunks = rag_engine.retrieve_relevant_chunks(user_query, user_id, top_k=4)
    
    retrieval_source = ""
    context = ""
    citations_expander_title = ""
    citations_content = ""
    
    if relevant_chunks:
        retrieval_source = "📄 Uploaded Documents"
        citations_expander_title = "📚 Retrieved Relevant Passages from Uploaded Documents"
        
        # Build context
        context_parts = []
        citations_parts = []
        for idx, chunk in enumerate(relevant_chunks):
            context_parts.append(f"[{chunk['document_name']} (Chunk {chunk['chunk_index']})]:\n{chunk['content']}")
            citations_parts.append(
                f"**Source: {chunk['document_name']} (Chunk {chunk['chunk_index']})** (BM25 Score: {chunk['score']:.2f})\n\n"
                f"{chunk['content']}\n\n---"
            )
        context = "\n\n".join(context_parts)
        citations_content = "\n\n".join(citations_parts)
        
    else:
        # Fallback to Tavily Web Search
        retrieval_source = "🌐 Web Search (Tavily Fallback)"
        citations_expander_title = "🌐 Tavily Search Engine Results"
        
        with st.spinner("Searching the web for answers..."):
            search_results, direct_ans = search_engine.web_search(user_query)
            
        if search_results:
            context_parts = []
            citations_parts = []
            for r in search_results:
                context_parts.append(f"Source URL: {r['url']}\nTitle: {r['title']}\nContent Snippet: {r['content']}")
                citations_parts.append(f"**[{r['title']}]({r['url']})**\n{r['content']}\n\n---")
            context = "\n\n".join(context_parts)
            citations_content = "\n\n".join(citations_parts)
        else:
            context = "No context found."
            citations_content = "No matching uploaded documents found, and Tavily web search was either unavailable or returned no results."
            
    # 4. Generate response from Groq
    with st.chat_message("assistant"):
        # Display source pill
        st.markdown(f"<div class='chat-bubble-source'>Source: {retrieval_source}</div>", unsafe_allow_html=True)
        
        # Assemble message thread for multi-turn history
        past_msgs = database.get_chat_messages(current_chat_id)
        # Limit history to last 6 turns (12 messages) to fit in context nicely
        history_buffer = []
        
        # We exclude the last user message because we'll format it with context
        for msg in past_msgs[:-1]:
            history_buffer.append({"role": msg["role"], "content": msg["content"]})
            
        # Build prompt instructions
        system_instruction = (
            "You are a helpful RAG AI Assistant. Answer the user's query utilizing the context provided below. "
            "If the context matches their query, answer based only on it. If the context is from Web Search (Tavily), "
            "synthesize a coherent answer from the search details.\n"
            "If you cannot find the answer in the provided context, state that clearly, then synthesize an answer "
            "using your general knowledge but make sure to flag it as general knowledge.\n"
            "Always maintain a polite, soft tone. Never write harmful content, violence, weapons advice, "
            "or expose sensitive user PII. If the user asks for harm, weapon creation, or private information, "
            "refuse politely (e.g. 'I cannot assist with requests involving harm, weapons, or private info').\n\n"
            f"=== RETRIEVED CONTEXT ({retrieval_source}) ===\n"
            f"{context}"
        )
        
        messages_payload = [
            {"role": "system", "content": system_instruction}
        ] + history_buffer + [
            {"role": "user", "content": user_query}
        ]
        
        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = groq_client.chat.completions.create(
                model=selected_model,
                messages=messages_payload,
                temperature=0.3,
                max_tokens=2048,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"Groq LLM Generation Error: {e}"
            response_placeholder.markdown(error_msg)
            full_response = error_msg
            
        # Display citations
        with st.expander(citations_expander_title):
            st.markdown(citations_content)
            
    # Save Assistant Response to Database
    database.add_message(current_chat_id, "assistant", full_response)
    st.rerun()
