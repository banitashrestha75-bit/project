import os
import sys

# Ensure project directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import database
import auth
import rag_engine
import guardrails
from logger_setup import logger

def run_tests():
    print("--- 1. INITIALIZING DATABASE SCHEMAS ---")
    try:
        database.init_db()
        print("[PASS] Database tables created successfully.")
    except Exception as e:
        print(f"[FAIL] Database initialization failed: {e}")
        return False

    print("\n--- 2. TESTING ADMIN AUTHENTICATION ---")
    admin_auth, msg, admin_data = auth.authenticate_admin("admin", "admin123")
    if admin_auth and admin_data and admin_data["username"] == "admin":
        print("[PASS] Seeded admin login verified.")
    else:
        print(f"[FAIL] Seeded admin login failed: {msg}")
        return False

    # Test bad admin password
    admin_auth_bad, _, _ = auth.authenticate_admin("admin", "wrong_pass")
    if not admin_auth_bad:
        print("[PASS] Bad admin password correctly rejected.")
    else:
        print("[FAIL] Admin login accepted wrong password!")
        return False

    print("\n--- 3. TESTING NORMAL USER REGISTRATION & AUTHORIZATION ---")
    test_email = "verif_user@test.com"
    test_pass = "VerifSecret123!"
    
    # Pre-clean
    conn = database.get_db_connection()
    conn.execute("DELETE FROM info WHERE email = ?", (test_email,))
    conn.commit()
    conn.close()
    
    # Register
    reg_success, reg_msg = auth.register_user(
        name="Test Verification User",
        contact="+977-9876543210",
        detail="Testing database profiles table",
        email=test_email,
        password=test_pass,
        address="Gaidakot, Nawalpur"
    )
    
    if reg_success and "successful" in reg_msg.lower():
        print("[PASS] Normal user registration succeeded.")
    else:
        print(f"[FAIL] User registration failed: {reg_msg}")
        return False
        
    # Check that they can login immediately
    login_success, login_msg, user_data = auth.authenticate_user(test_email, test_pass)
    if login_success and user_data and user_data["email"] == test_email:
        print("[PASS] Normal user login verified immediately after registration.")
    else:
        print(f"[FAIL] Login failed: {login_msg}")
        return False

    # Retrieve registered user to get ID
    new_user = database.get_user_by_email(test_email)
    if not new_user:
        print("[FAIL] Failed to retrieve registered user from database.")
        return False

    print("\n--- 4. TESTING RECURSIVE CHUNKER & COSINE SIMILARITY MATH ---")
    # Chunking
    sample_text = "This is sentence one of paragraph one. Sentence two of paragraph one.\n\nThis is paragraph two."
    chunks = rag_engine.chunk_text(sample_text, chunk_size=80, chunk_overlap=15)
    if chunks:
        print(f"[PASS] Text split into {len(chunks)} chunks successfully.")
    else:
        print("[FAIL] Chunking returned empty result.")
        return False
        
    # Cosine Similarity Math
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 1.0, 0.0]  # 45 degrees angle, cosine similarity should be 1/sqrt(2) = 0.707
    sim = rag_engine.cosine_similarity(v1, v2)
    if abs(sim - 0.7071) < 0.001:
        print(f"[PASS] Cosine similarity math matches target: {sim:.4f}")
    else:
        print(f"[FAIL] Cosine similarity math incorrect: {sim:.4f} (Expected: 0.7071)")
        return False

    print("\n--- 5. TESTING GUARDRAILS ---")
    # Test Greetings
    tg, resp = guardrails.handle_guardrails("hello assistant")
    if tg and "RAG" in resp:
        print("[PASS] Greeting guardrail triggered correctly.")
    else:
        print(f"[FAIL] Greeting guardrail failed: triggered={tg}, resp={resp}")
        return False
        
    # Test Self-Harm
    tg, resp = guardrails.handle_guardrails("I want to commit suicide")
    if tg and "helpline" in resp:
        print("[PASS] Self-harm guardrail triggered correctly.")
    else:
        print(f"[FAIL] Self-harm guardrail failed: triggered={tg}, resp={resp}")
        return False

    # Check for Groq API key to test embeddings
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        try:
            import streamlit as st
            if "GROQ_API_KEY" in st.secrets:
                groq_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
            
    if groq_key:
        print("\n--- 6. TESTING GROQ EMBEDDINGS & VECTOR SEARCH ---")
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            
            # Embed check
            emb = rag_engine.generate_single_embedding("hello", client)
            if len(emb) == 768:
                print(f"[PASS] Successfully generated 768-dim vector using nomic-embed-text-v1.5.")
            else:
                print(f"[FAIL] Embedding dimension was {len(emb)}, expected 768.")
                return False
                
            # Vector indexing
            test_chunks = [
                "Computer science is the study of algorithmic processes and computational machines.",
                "Hotel management involves overseeing operations in lodging and food services."
            ]
            test_embs = rag_engine.generate_embeddings(test_chunks, client)
            
            # Save vector chunks (global scope)
            database.add_vector_chunks(
                user_id=None,
                document_name="academic_disciplines.txt",
                chunks_list=test_chunks,
                embeddings_list=test_embs
            )
            print("[PASS] Global vector chunks indexed in database.")
            
            # Vector retrieval query
            results = rag_engine.retrieve_vector_chunks("What is computing and algorithms?", user_id=new_user["id"], client=client, top_k=1)
            if results and "Computer science" in results[0]["content"]:
                print(f"[PASS] Semantic vector search matched relevant chunk: '{results[0]['content'][:40]}...' (Score: {results[0]['score']:.3f})")
            else:
                print(f"[FAIL] Semantic vector search returned incorrect chunk: {results}")
                return False
                
            # Clean up vector chunks
            database.delete_user_chunks(None, "academic_disciplines.txt")
            print("[PASS] Cleanup vector chunks completed.")
        except Exception as e:
            print(f"[FAIL] Embeddings/Vector search execution failed: {e}")
            return False
    else:
        print("\n[INFO] GROQ_API_KEY not found in env. Skipping live embeddings & vector search checks.")

    # Cleanup Database entries
    conn = database.get_db_connection()
    conn.execute("DELETE FROM info WHERE id = ?", (new_user["id"],))
    conn.commit()
    conn.close()
    print("\n[PASS] Database cleanup completed.")
    
    print("\n===========================================")
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("===========================================")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
