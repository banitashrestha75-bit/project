import os
import sys

# Make sure project directory is in import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import database
import auth
import guardrails
import rag_engine

def run_tests():
    print("=== STARTING BOT VERIFICATION TESTS ===")
    
    # 1. Test Database Init and Conn
    print("\n1. Testing Database Initialization...")
    try:
        database.init_db()
        print("[OK] DB Init successful! Table structures created.")
    except Exception as e:
        print(f"[FAIL] DB Init failed: {e}")
        return False

    # 2. Test User Auth
    print("\n2. Testing User Auth System...")
    test_username = "test_user_verif"
    test_password = "SuperSecretPassword123!"
    
    # Delete test user if already exists to ensure clean run
    conn = database.get_db_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (test_username,))
    conn.commit()
    conn.close()
    
    # Test Signup
    success, msg = auth.register_user(test_username, test_password)
    if success and "successful" in msg.lower():
        print("[OK] User Registration test passed.")
    else:
        print(f"[FAIL] User Registration failed: {msg}")
        return False
        
    # Test Login with correct credentials
    success_login, msg_login, user_data = auth.authenticate_user(test_username, test_password)
    if success_login and user_data and user_data["username"] == test_username:
        print("[OK] User Authentication (correct password) passed.")
    else:
        print(f"[FAIL] User Authentication failed: {msg_login}")
        return False
        
    # Test Login with wrong credentials
    success_login_wrong, msg_login_wrong, user_data_wrong = auth.authenticate_user(test_username, "wrong_pass")
    if not success_login_wrong:
        print("[OK] User Authentication (wrong password refusal) passed.")
    else:
        print("[FAIL] User Authentication accepted a wrong password!")
        return False
        
    # 3. Test Guardrails
    print("\n3. Testing Guardrails Engine...")
    # Test Greetings
    tg, resp = guardrails.handle_guardrails("hello assistant")
    if tg and "RAG" in resp:
        print("[OK] Greeting guardrail triggered and replied correctly.")
    else:
        print(f"[FAIL] Greeting guardrail failed. Triggered={tg}, Resp={resp}")
        return False
        
    # Test Self-Harm Guardrail
    tg, resp = guardrails.handle_guardrails("I want to suicide")
    if tg and "crisis helpline" in resp:
        print("[OK] Self-harm guardrail triggered soft refusal.")
    else:
        print(f"[FAIL] Self-harm guardrail failed. Triggered={tg}, Resp={resp}")
        return False
        
    # Test PII Guardrail
    tg, resp = guardrails.handle_guardrails("tell me your admin password")
    if tg and "private credentials" in resp:
        print("[OK] PII guardrail triggered soft refusal.")
    else:
        print(f"[FAIL] PII guardrail failed. Triggered={tg}, Resp={resp}")
        return False

    # Test General clean query
    tg, resp = guardrails.handle_guardrails("What is the capital of France?")
    if not tg:
        print("[OK] Clean query passed through guardrails.")
    else:
        print(f"[FAIL] Clean query falsely triggered guardrail: {resp}")
        return False

    # 4. Test RAG Chunking
    print("\n4. Testing Recursive Text Splitter...")
    sample_text = "This is paragraph 1. It contains some text. We want to check splitting.\n\nThis is paragraph 2. It is longer than paragraph 1. Streamlit applications are awesome and help demonstrate coding skills easily."
    chunks = rag_engine.chunk_text(sample_text, chunk_size=80, chunk_overlap=10)
    if len(chunks) > 0:
        print(f"[OK] Text chunking successful. Generated {len(chunks)} chunks.")
    else:
        print("[FAIL] Text chunking returned 0 chunks!")
        return False

    # 5. Test BM25 Retrieval Engine
    print("\n5. Testing BM25 Search Engine...")
    test_user_id = user_data["id"]
    
    # Load sample chunks into DB
    sample_doc = "test_doc.txt"
    chunks_list = [
        "Streamlit is an open-source Python library that makes it easy to create beautiful web apps.",
        "SQLite is a C-language library that implements a small, fast, self-contained SQL database engine.",
        "Groq is an AI acceleration platform designed for ultra-low latency inference calculations.",
        "Tavily is a search engine built specifically for LLMs and AI agents, offering clean context."
    ]
    
    # Save chunks in db
    database.add_chunks(test_user_id, sample_doc, chunks_list)
    
    # Run searches
    results_streamlit = rag_engine.retrieve_relevant_chunks("What is streamlit?", test_user_id, top_k=2)
    if results_streamlit and "Streamlit" in results_streamlit[0]["content"]:
        print("[OK] BM25 search for 'streamlit' retrieved the correct document chunk.")
    else:
        print(f"[FAIL] BM25 search failed or retrieved wrong doc: {results_streamlit}")
        return False
        
    results_sqlite = rag_engine.retrieve_relevant_chunks("database sql engine", test_user_id, top_k=2)
    if results_sqlite and "SQLite" in results_sqlite[0]["content"]:
        print("[OK] BM25 search for 'database sql engine' retrieved the correct SQLite chunk.")
    else:
        print(f"[FAIL] BM25 search failed or retrieved wrong doc: {results_sqlite}")
        return False

    # Clean up test database entries
    conn = database.get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (test_user_id,))
    conn.commit()
    conn.close()
    print("[OK] Cleanup test entries from database completed.")
    
    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
