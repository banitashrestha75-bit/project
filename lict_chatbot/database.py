import sqlite3
import os
import json
import hashlib
from logger_setup import logger

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    logger.info("Initializing SQLite database schemas...")
    
    # 1. Create info table (normal users)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        contact TEXT NOT NULL,
        detail TEXT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        address TEXT NOT NULL,
        is_authorized INTEGER DEFAULT 0 -- 0: pending, 1: authorized
    );
    """)
    
    # 2. Create admin table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)
    
    # 3. Create chats table pointing to info
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES info(id) ON DELETE CASCADE
    );
    """)
    
    # 4. Create messages table pointing to chats
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
    );
    """)
    
    # 5. Create vector_chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vector_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, -- NULL means global/scraped document accessible to all, NOT NULL is user-specific
        document_name TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL, -- JSON string of floats
        metadata TEXT, -- JSON string metadata (e.g. url, page number)
        FOREIGN KEY(user_id) REFERENCES info(id) ON DELETE CASCADE
    );
    """)
    
    # Seed default admin user: admin / admin123 (plaintext password)
    cursor.execute("INSERT OR IGNORE INTO admin (username, password) VALUES (?, ?)", ("admin", "admin123"))
    
    # Seed dummy info record for admin (user_id=999999) to satisfy foreign key constraints in chats
    cursor.execute("""
    INSERT OR IGNORE INTO info (id, name, contact, detail, email, password_hash, address, is_authorized)
    VALUES (999999, 'Admin Account', '0000000000', 'Admin placeholder', 'admin@system.local', 'dummy', 'System', 1)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database schemas initialized, admin seeded, and dummy info seeded.")

# --- Admin Helpers ---
def get_admin(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# --- User Helpers (info table) ---
def create_user(name, contact, detail, email, password_hash, address, is_authorized=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute("""
        INSERT INTO info (name, contact, detail, email, password_hash, address, is_authorized)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, contact, detail, email.strip().lower(), password_hash, address, is_authorized))
        conn.commit()
        success = True
        logger.info(f"User created: {email} (Authorized: {is_authorized})")
    except sqlite3.IntegrityError as e:
        logger.warning(f"Failed to create user {email}: {e}")
    finally:
        conn.close()
    return success

def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM info WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM info WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM info ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_user_authorization(user_id, is_authorized):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE info SET is_authorized = ? WHERE id = ?", (is_authorized, user_id))
    conn.commit()
    conn.close()
    logger.info(f"Updated authorization for user ID {user_id} to {is_authorized}")

# --- Chat Helpers ---
def create_chat(chat_id, user_id, title):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (id, user_id, title) VALUES (?, ?, ?)", (chat_id, user_id, title))
    conn.commit()
    conn.close()
    logger.info(f"Created chat session: {chat_id} for user ID {user_id}")

def get_user_chats(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chats WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_chat(chat_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted chat session: {chat_id}")

# --- Message Helpers ---
def add_message(chat_id, role, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
    conn.commit()
    conn.close()

def get_chat_messages(chat_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- Vector Chunk Helpers ---
def add_vector_chunks(user_id, document_name, chunks_list, embeddings_list, metadata_list=None):
    """
    Saves chunks and their embedding vectors in the DB.
    user_id = None means global/scraped document.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Avoid duplicates: delete existing chunks for this document under this user scope
    if user_id is None:
        cursor.execute("DELETE FROM vector_chunks WHERE user_id IS NULL AND document_name = ?", (document_name,))
    else:
        cursor.execute("DELETE FROM vector_chunks WHERE user_id = ? AND document_name = ?", (user_id, document_name))
        
    for idx, (chunk, emb) in enumerate(zip(chunks_list, embeddings_list)):
        meta_str = json.dumps(metadata_list[idx]) if metadata_list else None
        emb_str = json.dumps(emb)
        cursor.execute("""
        INSERT INTO vector_chunks (user_id, document_name, chunk_index, content, embedding, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, document_name, idx, chunk, emb_str, meta_str))
        
    conn.commit()
    conn.close()
    logger.info(f"Indexed {len(chunks_list)} vector chunks for document '{document_name}' (User: {user_id})")

def get_user_vector_chunks(user_id):
    """Retrieves all vector chunks accessible to the user (global chunks + user's own chunks)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT * FROM vector_chunks WHERE user_id IS NULL")
    else:
        cursor.execute("SELECT * FROM vector_chunks WHERE user_id IS NULL OR user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    chunks = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding"])
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
        chunks.append(d)
    return chunks

def delete_user_chunks(user_id, document_name=None):
    """Deletes documents from the vector store."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if document_name:
        if user_id is None:
            cursor.execute("DELETE FROM vector_chunks WHERE user_id IS NULL AND document_name = ?", (document_name,))
        else:
            cursor.execute("DELETE FROM vector_chunks WHERE user_id = ? AND document_name = ?", (user_id, document_name))
        logger.info(f"Deleted document '{document_name}' from vector store (User: {user_id})")
    else:
        if user_id is None:
            cursor.execute("DELETE FROM vector_chunks WHERE user_id IS NULL")
        else:
            cursor.execute("DELETE FROM vector_chunks WHERE user_id = ?", (user_id,))
        logger.info(f"Deleted all documents from vector store (User: {user_id})")
    conn.commit()
    conn.close()

def get_user_uploaded_documents(user_id):
    """Returns a list of document names accessible to the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT DISTINCT document_name FROM vector_chunks WHERE user_id IS NULL")
    else:
        cursor.execute("SELECT DISTINCT document_name FROM vector_chunks WHERE user_id IS NULL OR user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row["document_name"] for row in rows]
