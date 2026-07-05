import hashlib
import database

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 with a predefined salt."""
    salt = "rag_chatbot_salt_2026_secured"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Checks if a password matches the stored hash."""
    return hash_password(password) == password_hash

def register_user(username, password):
    """Registers a new user by hashing password and saving in the database."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    
    # Check if user already exists
    existing_user = database.get_user(username)
    if existing_user:
        return False, "Username already exists."
    
    hashed = hash_password(password)
    success = database.create_user(username, hashed)
    if success:
        return True, "Registration successful! You can now log in."
    else:
        return False, "An error occurred during registration. Please try again."

def authenticate_user(username, password):
    """Authenticates a user against database records."""
    username = username.strip().lower()
    user = database.get_user(username)
    if not user:
        return False, "Username not found.", None
    
    if verify_password(password, user["password_hash"]):
        return True, "Login successful!", user
    else:
        return False, "Incorrect password.", None
