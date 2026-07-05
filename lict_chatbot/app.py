import streamlit as st
import os
import uuid
import dotenv
import pandas as pd
import json
from groq import Groq

# Load environment variables
dotenv.load_dotenv()
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
dotenv.load_dotenv(dotenv_path=env_path, override=False)

# Setup logger first
from logger_setup import logger, get_log_contents

# Import modules
import database
import auth
import parser
import rag_engine
import search_engine
import guardrails
import scraper

# Initialize SQLite database structure
database.init_db()

# --- Page Configurations ---
st.set_page_config(
    page_title="LICT RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ChatGPT-like Premium Dark Theme (CSS) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main View Area override */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0d !important;
        color: #e3e3e3 !important;
    }
    
    /* Header background transparent */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        width: 300px !important;
    }
    
    /* Sidebar text/titles */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: #171717;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0px 16px;
        color: #b4b4b4;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #212121 !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Form & Input Styling */
    .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div>div {
        background-color: #212121 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2f2f2f !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #3e3e3e !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Special primary button */
    .primary-btn button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
    }
    .primary-btn button:hover {
        background-color: #2563eb !important;
    }
    
    /* Custom ChatGPT User/Assistant bubble layouts */
    .chat-bubble-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        max-width: 800px;
        margin: 0 auto 100px auto;
        padding-top: 20px;
    }
    
    .chat-msg {
        display: flex;
        padding: 1rem;
        border-radius: 12px;
        gap: 16px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    
    .chat-msg.user {
        background-color: #212121;
    }
    
    .chat-msg.assistant {
        background-color: #171717;
    }
    
    .avatar-icon {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .avatar-icon.user {
        background-color: #4f46e5;
        color: white;
    }
    
    .avatar-icon.assistant {
        background-color: #10b981;
        color: white;
    }
    
    .msg-content {
        flex: 1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .source-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        margin-top: 6px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Dataframe backgrounds */
    [data-testid="stTable"] table, [data-testid="stDataFrame"] {
        background-color: #171717 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
    }
    
    /* Main Layout Headers */
    .gpt-header {
        text-align: center;
        margin: 3rem 0;
    }
    .gpt-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .gpt-subtitle {
        color: #8e8e93;
        font-size: 1rem;
        margin-top: 0.5rem;
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
if "role" not in st.session_state:
    st.session_state.role = None  # 'admin' or 'user'
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "temp_chunks" not in st.session_state:
    st.session_state.temp_chunks = []
if "temp_doc_name" not in st.session_state:
    st.session_state.temp_doc_name = None
if "active_view" not in st.session_state:
    st.session_state.active_view = "Chat"

# --- USER AUTHENTICATION SCREEN ---
if not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 1.5, 1])
    with col_m:
        st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: white;'>🤖 AI RAG Chatbot Portal</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8e8e93; margin-bottom: 2rem;'>Access local knowledge base & web search fallback</p>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔒 User Sign In", "📝 User Registration", "🛠️ Admin Portal"])
        
        with tab1:
            with st.form("user_login_form"):
                email = st.text_input("Email").strip()
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit:
                    success, msg, user_data = auth.authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = user_data
                        st.session_state.role = "user"
                        st.session_state.active_view = "Chat"
                        logger.info(f"User logged in: {email}")
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with tab2:
            with st.form("user_registration_form"):
                st.info("Registration details will save in the secure database. You must be authorized by an admin before signing in.")
                reg_name = st.text_input("Full Name *")
                reg_contact = st.text_input("Contact Number *")
                reg_email = st.text_input("Email Address *")
                reg_address = st.text_input("Address *")
                reg_detail = st.text_area("Purpose/Details (Optional)")
                reg_password = st.text_input("Password *", type="password")
                reg_confirm = st.text_input("Confirm Password *", type="password")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_reg:
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = auth.register_user(
                            name=reg_name,
                            contact=reg_contact,
                            detail=reg_detail,
                            email=reg_email,
                            password=reg_password,
                            address=reg_address
                        )
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                            
        with tab3:
            with st.form("admin_login_form"):
                admin_username = st.text_input("Admin Username").strip()
                admin_password = st.text_input("Admin Password", type="password")
                submit_admin = st.form_submit_button("Admin Log In", use_container_width=True)
                
                if submit_admin:
                    success, msg, admin_data = auth.authenticate_admin(admin_username, admin_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = admin_data
                        st.session_state.role = "admin"
                        st.session_state.active_view = "Admin Dashboard"
                        logger.info(f"Admin logged in: {admin_username}")
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                        
    st.stop()

# --- LOGGED IN ROUTING SYSTEM ---
user_data = st.session_state.user
role = st.session_state.role
is_admin = (role == "admin")

# Resolve variables depending on login type
if is_admin:
    user_id = 999999  # Dummy ID for Admin global uploads
    display_name = f"Admin: {user_data['username']}"
else:
    user_id = user_data["id"]
    display_name = user_data["name"]

groq_client = get_groq_client()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown(f"### 👋 {display_name}")
    st.caption(f"Role: {role.upper()}")
    
    if st.button("🚪 Log Out", use_container_width=True):
        logger.info(f"Logout triggered for user: {display_name}")
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.current_chat_id = None
        st.session_state.temp_chunks = []
        st.session_state.temp_doc_name = None
        st.session_state.active_view = "Chat"
        st.rerun()
        
    st.markdown("---")
    
    # Navigation Views
    st.markdown("### 🧭 Navigation")
    
    # Navigation buttons
    if st.button("💬 Conversational Chat", use_container_width=True):
        st.session_state.active_view = "Chat"
        st.rerun()
        
    if st.button("🗃️ Vector DB Inspector", use_container_width=True):
        st.session_state.active_view = "Vector DB"
        st.rerun()
        
    if is_admin:
        if st.button("🛡️ Admin Dashboard", use_container_width=True):
            st.session_state.active_view = "Admin Dashboard"
            st.rerun()
            
    st.markdown("---")
    
    # Chat Parameters
    st.markdown("### 🧠 Configuration")
    
    selected_model = st.selectbox(
        "LLM Model",
        options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    retrieval_mode = st.selectbox(
        "Search Algorithm",
        options=["Vector Similarity", "BM25 Keyword Matching"],
        index=0
    )
    
    st.markdown("---")
    
    # Chat Sessions
    if st.session_state.active_view == "Chat":
        st.markdown("### 💬 Chat History")
        if st.button("➕ New Chat Session", use_container_width=True):
            chat_id = str(uuid.uuid4())
            database.create_chat(chat_id, user_id, "New Chat Session")
            st.session_state.current_chat_id = chat_id
            st.rerun()
            
        chats = database.get_user_chats(user_id)
        if not chats:
            st.caption("No active chat sessions. Start one above!")
        else:
            if st.session_state.current_chat_id is None:
                st.session_state.current_chat_id = chats[0]["id"]
                
            for chat in chats:
                col1, col2 = st.columns([0.85, 0.15])
                is_active = (chat["id"] == st.session_state.current_chat_id)
                # Bold highlight active chat
                btn_lbl = f"▶ {chat['title']}" if is_active else chat["title"]
                
                with col1:
                    if st.button(btn_lbl, key=f"sel_{chat['id']}", use_container_width=True, help=chat["title"]):
                        st.session_state.current_chat_id = chat["id"]
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{chat['id']}", help="Delete Chat"):
                        database.delete_chat(chat["id"])
                        if st.session_state.current_chat_id == chat["id"]:
                            st.session_state.current_chat_id = None
                        st.rerun()
                        
        st.markdown("---")

# --- MAIN APP LAYOUT ROUTER ---

# CHECK GROQ CLIENT
if not groq_client:
    st.error("⚠️ GROQ_API_KEY is not configured! Please configure it in your .env file or Streamlit Secrets. The chat interface is disabled.")
    st.stop()


# A. VIEW: CHAT INTERFACE
if st.session_state.active_view == "Chat":
    st.markdown("<div class='gpt-header'><div class='gpt-title'>🤖 Conversational Assistant</div><div class='gpt-subtitle'>Query vector indexes or fall back to Tavily Web Search</div></div>", unsafe_allow_html=True)
    
    # Chat Dialogue Container
    if st.session_state.current_chat_id is None:
        chat_id = str(uuid.uuid4())
        database.create_chat(chat_id, user_id, "New Chat Session")
        st.session_state.current_chat_id = chat_id
        st.rerun()
        
    current_chat_id = st.session_state.current_chat_id
    messages = database.get_chat_messages(current_chat_id)
    
    # Display Chat Bubbles
    st.markdown("<div class='chat-bubble-container'>", unsafe_allow_html=True)
    for msg in messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        role_label = "U" if msg["role"] == "user" else "AI"
        
        st.markdown(f"""
        <div class="chat-msg {role_class}">
            <div class="avatar-icon {role_class}">{role_label}</div>
            <div class="msg-content">
                {msg['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom input box styled like ChatGPT
    user_query = st.chat_input("Message RAG assistant...")
    
    if user_query:
        user_query = user_query.strip()
        
        # Save user message
        database.add_message(current_chat_id, "user", user_query)
        
        # Rename chat session if still default
        chat_title = "New Chat Session"
        chats = database.get_user_chats(user_id)
        for c in chats:
            if c["id"] == current_chat_id:
                chat_title = c["title"]
                break
        if chat_title == "New Chat Session":
            new_title = user_query[:35] + ("..." if len(user_query) > 35 else "")
            conn = database.get_db_connection()
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, current_chat_id))
            conn.commit()
            conn.close()
            
        # Apply Guardrails Pre-processing
        guard_triggered, guard_response = guardrails.handle_guardrails(user_query)
        if guard_triggered:
            database.add_message(current_chat_id, "assistant", guard_response)
            logger.info("Guardrail triggered on query.")
            st.rerun()
            
        # Perform retrieval
        with st.spinner("Querying vector database..."):
            if retrieval_mode == "Vector Similarity":
                relevant_chunks = rag_engine.retrieve_vector_chunks(user_query, user_id, groq_client, top_k=4)
            else:
                relevant_chunks = rag_engine.retrieve_bm25_chunks(user_query, user_id, top_k=4)
                
        retrieval_source = ""
        context = ""
        citations = []
        
        if relevant_chunks:
            retrieval_source = "Vector DB" if retrieval_mode == "Vector Similarity" else "BM25 Match"
            context_parts = []
            for idx, chunk in enumerate(relevant_chunks):
                doc_name = chunk["document_name"]
                c_idx = chunk["chunk_index"]
                context_parts.append(f"[{doc_name} (Chunk {c_idx})]:\n{chunk['content']}")
                citations.append(f"**[{idx+1}] Source: {doc_name} (Chunk {c_idx})** (Score: {chunk['score']:.3f})\n\n{chunk['content']}")
            context = "\n\n".join(context_parts)
        else:
            # Fallback to Tavily
            retrieval_source = "Tavily Web Search"
            with st.spinner("Searching the web..."):
                search_results, direct_ans = search_engine.web_search(user_query)
                
            if search_results:
                context_parts = []
                for idx, r in enumerate(search_results):
                    context_parts.append(f"URL: {r['url']}\nTitle: {r['title']}\nSnippet: {r['content']}")
                    citations.append(f"**[{idx+1}] [{r['title']}]({r['url']})**\n{r['content']}")
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant context found."
                citations.append("No database matching document chunks found and web search is empty/disabled.")

        # Build final system message
        system_instruction = (
            "You are a helpful RAG AI Assistant. Answer the user's query utilizing the context provided below. "
            "If the context matches their query, answer based only on it. If the context is from Web Search (Tavily), "
            "synthesize a coherent answer from the search details.\n"
            "If you cannot find the answer in the provided context, state that clearly, then synthesize an answer "
            "using your general knowledge but make sure to flag it as general knowledge.\n"
            "Always maintain a polite, soft tone. Never write harmful content, violence, weapons advice, "
            "or expose sensitive user PII. If the user asks for harm, weapon creation, or private information, "
            "refuse politely (e.g. 'I cannot assist with requests involving harm, weapons, or private info').\n\n"
            f"=== RETRIEVED CONTEXT (Source: {retrieval_source}) ===\n"
            f"{context}"
        )
        
        # Retrieve history
        past_msgs = database.get_chat_messages(current_chat_id)[:-1]
        history_buffer = [{"role": m["role"], "content": m["content"]} for m in past_msgs]
        
        messages_payload = [
            {"role": "system", "content": system_instruction}
        ] + history_buffer + [
            {"role": "user", "content": user_query}
        ]
        
        # Call Groq to generate assistant response
        with st.chat_message("assistant"):
            st.markdown(f"<div class='source-pill'>Source: {retrieval_source}</div>", unsafe_allow_html=True)
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
                error_msg = f"Groq Generation Error: {e}"
                response_placeholder.markdown(error_msg)
                full_response = error_msg
                
            # Render citations
            with st.expander("📚 Sources & References", expanded=False):
                st.markdown("\n\n---\n\n".join(citations))
                
        # Save response in SQLite
        database.add_message(current_chat_id, "assistant", full_response)
        st.rerun()


# B. VIEW: VECTOR DB INSPECTOR
elif st.session_state.active_view == "Vector DB":
    st.markdown("<div class='gpt-header'><div class='gpt-title'>🗃️ Vector DB Inspector</div><div class='gpt-subtitle'>Explore text chunks and test semantic similarity searches directly</div></div>", unsafe_allow_html=True)
    
    st.write("This tool queries the SQLite database directly and displays chunks alongside their vector embeddings generated by `nomic-embed-text-v1.5`.")
    
    # Load all chunks for this user (including global scraped data)
    chunks = database.get_user_vector_chunks(user_id)
    
    tab_list, tab_search = st.tabs(["📋 Indexed Chunks List", "🔍 Semantic Query Sandbox"])
    
    with tab_list:
        if not chunks:
            st.warning("No document chunks exist in the vector database scope. Go to the Chat tab and upload files or trigger scraper in the Admin panel.")
        else:
            st.markdown(f"**Total chunks in database scope:** `{len(chunks)}`")
            
            rows = []
            for c in chunks:
                emb_prev = str(c["embedding"][:5])[:-1] + ", ...]"
                rows.append({
                    "ID": c["id"],
                    "Document Name": c["document_name"],
                    "Index": c["chunk_index"],
                    "Content Snippet": c["content"][:100] + "..." if len(c["content"]) > 100 else c["content"],
                    "Vector Dimension": len(c["embedding"]),
                    "Embedding Preview": emb_prev,
                    "Access Type": "Global" if c["user_id"] is None else "Private User File"
                })
                
            df_chunks = pd.DataFrame(rows)
            st.dataframe(df_chunks, use_container_width=True)
            
    with tab_search:
        st.markdown("#### Test Cosine Similarity Math against Database Chunks")
        query_text = st.text_input("Enter a query to test embedding cosine similarity similarity:")
        
        if query_text:
            query_text = query_text.strip()
            with st.spinner("Generating query embedding & calculating distances..."):
                query_emb = rag_engine.generate_single_embedding(query_text, groq_client)
                
                results = []
                for c in chunks:
                    sim = rag_engine.cosine_similarity(query_emb, c["embedding"])
                    results.append({
                        "Similarity Score": sim,
                        "Document": c["document_name"],
                        "Chunk Index": c["chunk_index"],
                        "Content Match": c["content"],
                        "Vector Preview": str(c["embedding"][:4])[:-1] + ", ...]"
                    })
                    
                # Sort by score
                results.sort(key=lambda x: x["Similarity Score"], reverse=True)
                df_res = pd.DataFrame(results)
                
                if df_res.empty:
                    st.info("No vector data to search.")
                else:
                    st.success("Similarity search finished!")
                    st.dataframe(df_res, use_container_width=True)


# C. VIEW: ADMIN DASHBOARD (ADMIN ONLY)
elif st.session_state.active_view == "Admin Dashboard" and is_admin:
    st.markdown("<div class='gpt-header'><div class='gpt-title'>🛡️ Admin Portal</div><div class='gpt-subtitle'>Review user registries, control scraping, and inspect real-time system logs</div></div>", unsafe_allow_html=True)
    
    tab_users, tab_scrape, tab_logs = st.tabs(["👥 User Authorization", "🌐 LICT College Crawler", "📜 Live Application Logs"])
    
    with tab_users:
        st.markdown("#### Registered User Registry (info table)")
        users = database.get_all_users()
        
        if not users:
            st.info("No users registered in the database.")
        else:
            # We want to show detailed info and an action button to toggle authorization
            df_users = pd.DataFrame(users)
            # Reorder fields for nice visual layout
            cols = ["id", "name", "email", "contact", "address", "detail", "is_authorized"]
            df_users = df_users[cols]
            
            st.dataframe(df_users, use_container_width=True)
            
            st.markdown("##### Change User Authorization Status")
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                selected_user_id = st.selectbox(
                    "Select User to Authorize/Deauthorize",
                    options=[u["id"] for u in users],
                    format_func=lambda x: next(f"ID {u['id']}: {u['name']} ({u['email']})" for u in users if u["id"] == x)
                )
            
            with col2:
                selected_user = next(u for u in users if u["id"] == selected_user_id)
                current_status = selected_user["is_authorized"]
                btn_txt = "Deauthorize User" if current_status == 1 else "Authorize User"
                
                if st.button(btn_txt, use_container_width=True):
                    new_status = 0 if current_status == 1 else 1
                    database.update_user_authorization(selected_user_id, new_status)
                    st.success(f"Successfully toggled status for {selected_user['name']} to {'Authorized' if new_status==1 else 'Pending'}.")
                    st.rerun()
                    
            # Helper to auto authorize everyone for quick grading/testing
            if st.button("🔓 Auto-Authorize All Pending Users (Quick Grading Dev Feature)"):
                for u in users:
                    if u["is_authorized"] == 0:
                        database.update_user_authorization(u["id"], 1)
                st.success("Authorized all pending users!")
                st.rerun()

    with tab_scrape:
        st.markdown("#### Crawl and Index Lumbini ICT College Website (`https://lict.edu.np/`)")
        st.write("Running this starts the BS4 crawler to recursively download text from the college website, formats the raw text into structured JSON (`lict_data.json`), chunks it, generates vector embeddings via Groq API, and saves it in the database under global scope (accessible to all accounts).")
        
        max_p = st.number_input("Maximum pages to crawl", min_value=2, max_value=60, value=25, step=1)
        
        if st.button("🚀 Trigger Scraper & Generate Vector Embeddings", type="primary"):
            status_container = st.empty()
            log_container = st.empty()
            
            logger.info("Triggered admin scrape for Lumbini ICT College site...")
            
            with st.spinner("Scraping Lumbini ICT College site..."):
                status_container.info("Step 1/3: Crawling site pages...")
                
                temp_dir = os.path.dirname(os.path.abspath(__file__))
                json_path = os.path.join(temp_dir, "lict_data.json")
                
                # Instantiate scraper with requested page limit
                sc = scraper.LICTScraper(max_pages=max_p)
                scraped_pages = sc.crawl()
                
                # Save structured JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(scraped_pages, f, indent=2, ensure_ascii=False)
                    
                status_container.info(f"Step 2/3: Saved {len(scraped_pages)} pages to raw JSON structure `lict_data.json`!")
                logger.info(f"Scraped {len(scraped_pages)} pages and saved raw JSON.")
                
            with st.spinner("Processing text and generating embeddings via Groq..."):
                status_container.info("Step 3/3: Generating embeddings & loading to Vector database...")
                
                total_chunks_saved = 0
                
                # Chunk and embed each page
                for page in scraped_pages:
                    url = page["url"]
                    title = page["title"]
                    body = page["text"]
                    
                    # Skip empty content
                    if not body.strip():
                        continue
                        
                    chunks = rag_engine.chunk_text(body)
                    if not chunks:
                        continue
                        
                    embeddings = rag_engine.generate_embeddings(chunks, groq_client)
                    
                    # Store chunks in Vector Store with user_id = None (Global Scope)
                    # Include source URL in metadata
                    metadata_list = [{"url": url, "title": title, "type": "scraped_college"}] * len(chunks)
                    
                    database.add_vector_chunks(
                        user_id=None,
                        document_name=title,
                        chunks_list=chunks,
                        embeddings_list=embeddings,
                        metadata_list=metadata_list
                    )
                    total_chunks_saved += len(chunks)
                    
                status_container.success(f"✅ Scraping completed! Crawled {len(scraped_pages)} pages, generated {total_chunks_saved} chunks and loaded them into SQLite Vector DB.")
                logger.info(f"Index complete: {total_chunks_saved} chunks loaded.")
                
            # Display Scraped JSON preview
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    preview_data = json.load(f)
                st.markdown("##### 📁 Preview Scraped Structured JSON (`lict_data.json`)")
                st.json(preview_data[:2])

    with tab_logs:
        st.markdown("#### Real-time Log Stream")
        st.write("This displays recent events logged by the RAG system to `app.log`.")
        
        if st.button("🔄 Refresh Logs"):
            st.rerun()
            
        logs_content = get_log_contents(100)
        st.code(logs_content, language="text")
