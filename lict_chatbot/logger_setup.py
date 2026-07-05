import logging
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")

def setup_logger():
    """Sets up a logger that logs to both the console and a file."""
    logger = logging.getLogger("rag_chatbot")
    
    # If logger is already configured, don't add duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def get_log_contents(max_lines: int = 100) -> str:
    """Reads the last N lines of the log file."""
    if not os.path.exists(LOG_FILE):
        return "No log file found."
        
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Error reading logs: {e}"
