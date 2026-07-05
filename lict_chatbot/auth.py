import hashlib
import database
from logger_setup import logger

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 with a predefined salt."""
    salt = "rag_chatbot_salt_2026_secured"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Checks if a password matches the stored hash."""
    return hash_password(password) == password_hash

def register_user(name, contact, detail, email, password, address):
    """Registers a new user inside the info table."""
    name = name.strip()
    contact = contact.strip()
    detail = detail.strip()
    email = email.strip().lower()
    address = address.strip()
    
    if not name or not contact or not email or not password or not address:
        return False, "All required fields (Name, Contact, Email, Password, Address) must be filled."
        
    # Check if user already exists
    existing_user = database.get_user_by_email(email)
    if existing_user:
        return False, "An account with this email already exists."
        
    hashed = hash_password(password)
    # Default is_authorized to 1 (instantly authorized as requested by user)
    success = database.create_user(name, contact, detail, email, hashed, address, is_authorized=1)
    if success:
        logger.info(f"User registration request received: {email}")
        return True, "Registration successful! You can now log in."
    else:
        return False, "An error occurred during registration. Please try again."

def authenticate_user(email, password):
    """Authenticates a normal user against the info table records."""
    email = email.strip().lower()
    user = database.get_user_by_email(email)
    if not user:
        return False, "Account with this email not found.", None
        
    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect password.", None
        
    logger.info(f"User login successful: {email}")
    return True, "Login successful!", user

def authenticate_admin(username, password):
    """Authenticates an administrator against the admin table records."""
    username = username.strip()
    admin = database.get_admin(username)
    if not admin:
        return False, "Admin account not found.", None
        
    if admin["password"] != password:
        return False, "Incorrect admin password.", None
        
    logger.info(f"Admin login successful: {username}")
    return True, "Admin login successful!", admin
