import os
import re
import sqlite3
import logging
import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from llama_parse import LlamaParse
from tavily import TavilyClient
from dotenv import load_dotenv

# ---------------------------------------------------------
# Automatically Load Environment Variables from Parent Folder (.env)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_env_path = os.path.abspath(os.path.join(current_dir, "..", ".env"))

# Try multiple possible .env locations
env_paths = [
    parent_env_path,
    os.path.join(current_dir, ".env"),
    os.path.join(os.path.dirname(current_dir), ".env")
]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        print(f"Loaded .env from: {env_path}")
        break

if not env_loaded:
    print("WARNING: No .env file found in expected locations!")

# Debug: Print all environment variables (for troubleshooting - remove in production)
print("Environment variables found:")
print(f"GROQ_API_KEY: {'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
#print(f"LLAMA_CLOUD_API_KEY: {'SET' if os.getenv('LLAMA_CLOUD_API_KEY') else 'NOT SET'}")
print(f"LLAMA_PARSE_API_KEY: {'SET' if os.getenv('LLAMA_PARSE_API_KEY') else 'NOT SET'}")
print(f"TAVILY_API_KEY: {'SET' if os.getenv('TAVILY_API_KEY') else 'NOT SET'}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# This will accept ANY of the common ways you might have written it in your .env file
LLAMA_API_KEY = (
    os.getenv("LLAMA_CLOUD_API_KEY") or 
    os.getenv("LLAMA_PARSE_KEY") or 
    os.getenv("LLAMA_API_KEY") or
    os.getenv("llama_parse_api_key") or
    os.getenv("LLAMAPARSE_API_KEY")  # Added another common variation
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# SQLite Database Setup & User Authentication
# ---------------------------------------------------------
DB_FILE = "users_info.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            contact TEXT,
            detail TEXT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            address TEXT,
            role TEXT DEFAULT 'user'
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM info")
    if cursor.fetchone()[0] == 0:
        logger.info("Database empty. Seeding initial authorized administrators and users.")
        users = [
            ('Admin User', '+977-9855078531', 'System Administrator', 'admin@lict.edu.np', 'admin123', 'Gaindakot-4, Nawalparasi', 'admin'),
            ('Ram Bahadur', '+977-78502188', 'BSc CSIT Faculty', 'ram@lict.edu.np', 'password123', 'Gaindakot, Nawalpur', 'user'),
            ('Sita Thapa', '+977-78503206', 'BCA Student', 'sita@lict.edu.np', 'sita2026', 'Narayangarh, Chitwan', 'user')
        ]
        cursor.executemany("""
            INSERT INTO info (name, contact, detail, email, password, address, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, users)
        conn.commit()
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, role FROM info WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user

# ---------------------------------------------------------
# Scraping & Structuring JSON using LlamaParse
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def scrape_and_parse_data():
    logger.info("Starting scrape for Lumbini ICT Campus...")
    url = "https://lict.edu.np/"
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_text = soup.get_text(separator="\n")
        else:
            raise Exception(f"Status code {response.status_code}")
    except Exception as e:
        logger.warning(f"Live scrape failed: {e}. Utilizing comprehensive pre-fetched backup content.")
        raw_text = """
        Lumbini ICT Campus (LICTC) is a premier Tribhuvan University (TU) affiliated private institution 
        located at Gaindakot-4, Kaligandaki Chowk, Nawalparasi (Nawalpur), Gandaki Province, Nepal. Established in 2013 AD (2069 BS).
        Programs offered: 
        1. BSc CSIT (Bachelor of Science in Computer Science and Information Technology) - 4 Years, 8 Semesters, 48 seats.
        2. BCA (Bachelor of Computer Application) - 4 Years, 8 Semesters, 35 seats.
        3. BITM / BIM (Bachelor of Information Technology Management) - 4 Years.
        4. BHM (Bachelor of Hotel Management) - 4 Years.
        Contact Info: Phone: +977-78-502188, +977-78-503206. Email: mail@lict.edu.np, lumbiniictcollege@gmail.com.
        Chairman: Kailash Koirala. Principal: Pratap Koirala.
        """

    cleaned_text = re.sub(r'\n+', '\n', raw_text).strip()
    temp_file = "raw_campus_data.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    logger.info("Passing unstructured data to LlamaParse...")
    try:
        # Check if API key is available before proceeding
        if not LLAMA_API_KEY:
            logger.error("LlamaParse API key is missing!")
            return [{"chunk_id": 0, "source": "Scraper Raw", "content": cleaned_text}]
            
        parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
        documents = parser.load_data(temp_file)
        
        chunks_json = []
        for idx, doc in enumerate(documents):
            chunks_json.append({
                "chunk_id": idx,
                "source": "Lumbini ICT Campus Web Docs",
                "content": doc.text
            })
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        logger.info("Successfully produced structured JSON chunks via LlamaParse.")
        return chunks_json
    except Exception as parse_err:
        logger.error(f"LlamaParse error: {parse_err}")
        return [{"chunk_id": 0, "source": "Scraper Raw", "content": cleaned_text}]

# ---------------------------------------------------------
# Core Assistant Engine (Groq + RAG Context + Tavily)
# ---------------------------------------------------------
def generate_ai_response(user_query, chunks):
    lowered_query = user_query.lower().strip()
    
    if lowered_query in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]:
        return "Hello! Welcome to the Lumbini ICT Campus AI Portal. How can I assist you today? 🎓"

    context_str = ""
    for chunk in chunks:
        context_str += f"\n--- Section: {chunk['source']} ---\n{chunk['content']}\n"

    web_context = ""
    if any(keyword in lowered_query for keyword in ["news", "event", "current", "date", "fee", "admission"]):
        try:
            logger.info(f"Triggering Tavily Web Query: {user_query}")
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            web_search = tavily.search(query=f"Lumbini ICT Campus {user_query}", max_results=2)
            web_context = "\n".join([res['content'] for res in web_search.get('results', [])])
        except Exception as e:
            logger.warning(f"Tavily lookup bypassed: {e}")

    system_prompt = f"""
    You are the Lumbini ICT AI Assistant, the official Executive AI Knowledge Chatbot for Lumbini ICT Campus. 
    Provide highly accurate, professional, and crisp responses about the institution.
    
    Internal verified institution database:
    {context_str}
    
    Real-time context:
    {web_context}
    
    Always remain concise, welcoming, and academically distinguished.
    """

    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_query})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )
        return completion.choices[0].message.content
    except Exception as err:
        logger.error(f"Groq API Processing Exception: {err}")
        return "My apologies, I am encountering connectivity drops processing your request. Please check systemic backend API configuration keys."

# ---------------------------------------------------------
# Streamlit Interface Implementation
# ---------------------------------------------------------
st.set_page_config(page_title="Lumbini ICT AI Assistant", page_icon="🤖", layout="wide")
init_db()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "parsed_database" not in st.session_state:
    st.session_state.parsed_database = None

# Sidebar Setup
with st.sidebar:
    st.title("🛡️ Secure Access Terminal")
    st.markdown("---")
    
    if not st.session_state.authenticated:
        st.subheader("Login Authorization")
        login_email = st.text_input("Institutional Email", placeholder="user@lict.edu.np")
        login_pass = st.text_input("Secure Password", type="password")
        
        if st.button("Authenticate Login", use_container_width=True):
            user = authenticate_user(login_email, login_pass)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_info = {"name": user[0], "email": user[1], "role": user[2]}
                st.success(f"Welcome, {user[0]}!")
                st.rerun()
            else:
                st.error("Access Denied: Credentials not authorized within Campus SQLite Engine.")
    else:
        st.markdown(f"### 👤 Active Account")
        st.info(f"**Name:** {st.session_state.user_info['name']}\n\n**Role:** {st.session_state.user_info['role'].upper()}")
        st.markdown("---")
        if st.button("Log Out Securely", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.chat_history = []
            st.rerun()

# Application Workspace Routing
if not st.session_state.authenticated:
    st.info("👋 **Welcome to the Lumbini ICT Campus AI Portal.** Please utilize your authorization credentials via the secure side-panel terminal input to begin.")
else:
    # SMART KEY CHECK: Pinpoint exactly which key isn't reading correctly
    missing_keys = []
    if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")
    if not LLAMA_API_KEY: missing_keys.append("LLAMA_CLOUD_API_KEY (or LLAMA_PARSE_API_KEY)")
    if not TAVILY_API_KEY: missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        st.error(f"⚠️ **Key Configuration Error!** Your `.env` file was found, but the following specific variables inside it are either completely missing, empty, or misspelled:\n\n" + 
                 "\n".join([f"* `{key}`" for key in missing_keys]) + 
                 "\n\nPlease check your configuration file and make sure there are no spaces around the `=` signs.")
    else:
        if st.session_state.parsed_database is None:
            with st.spinner("Compiling and structuring Lumbini ICT Campus web knowledge graphs via LlamaParse..."):
                st.session_state.parsed_database = scrape_and_parse_data()
                st.success("Ingestion Pipeline complete! Data chunks ready.")

        tab_chat, tab_vector = st.tabs(["💬 Lumbini ICT AI Assistant", "💾 Vector Database Chunks Inspect Tool"])

        with tab_chat:
            st.title("🤖 Lumbini ICT AI Assistant")
            st.caption("Context-Aware RAG Engine Powered by Llama3, LlamaParse, and Tavily Core Intelligence.")
            
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if prompt := st.chat_input("Ask anything regarding Lumbini ICT Campus curriculum or updates..."):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        response = generate_ai_response(prompt, st.session_state.parsed_database)
                        st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        with tab_vector:
            st.title("📦 Parsed Structured Data Chunks")
            st.json(st.session_state.parsed_database)