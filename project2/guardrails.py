import re

# Comprehensive list of greeting patterns (compiled regex)
GREETINGS_PATTERNS = [
    r"\b(hi|hello|hey|howdy|hola|yo|greetings|aloha|shalom)\b",
    r"\bhow are you\b",
    r"\bwhat'?s up\b",
    r"\bhow'?s it going\b",
    r"\b(good morning|good afternoon|good evening)\b"
]

# Compile greeting regex
greetings_re = re.compile("|".join(GREETINGS_PATTERNS), re.IGNORECASE)

# Sensitive and harmful topic keywords
SENSITIVE_PATTERNS = [
    # Violence, killing, harm, self-harm, weapons
    r"\b(kill|murder|slay|assassinate|suicide|self-harm|hurt myself|end my life|die)\b",
    r"\b(bomb|explosive|grenade|gun|rifle|pistol|weapon|shoot|poison|torture)\b",
    # Hacking and illegal activities
    r"\b(hack|cyberattack|ddos|exploit|phish|malware|ransomware|steal|rob|plagiarize)\b",
    # Private / PII data request (passwords, SSNs, credit cards, credentials)
    r"\b(password|credit card|social security|ssn|private key|api key|credentials|passcode|cvv)\b"
]

# Compile sensitive regex
sensitive_re = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)

def check_greetings(query: str) -> str:
    """Checks if the query is a simple greeting and returns a polite reply, or None."""
    query = query.strip()
    # Check if the query is relatively short and matches a greeting pattern
    if len(query) < 40 and greetings_re.search(query):
        return (
            "Hello! I am your RAG AI Assistant. I can help you query, search, and "
            "summarize your uploaded documents, or search the web if needed. How can I help you today?"
        )
    return None

def check_sensitive_topics(query: str) -> str:
    """Checks if the query contains sensitive/harmful topics and returns a soft refusal, or None."""
    if sensitive_re.search(query):
        # Determine specific message depending on the keywords triggered
        query_lower = query.lower()
        if any(w in query_lower for w in ["password", "ssn", "credit card", "private key", "credentials"]):
            return (
                "I cannot assist with requests involving private credentials, passwords, "
                "social security numbers, or sensitive personal information. Please keep your personal data secure."
            )
        elif any(w in query_lower for w in ["suicide", "self-harm", "hurt myself", "end my life"]):
            return (
                "If you are feeling overwhelmed and considering self-harm, please know that you are not alone. "
                "Please reach out to a crisis helpline (such as 988 in the US/Canada or 111 in the UK) "
                "or contact a professional. I cannot assist with topics related to self-harm."
            )
        else:
            return (
                "I cannot assist with queries related to violence, weapons, harm, hacking, "
                "or other illegal activities. Let me know if you have questions about your uploaded documents or other general topics."
            )
    return None

def handle_guardrails(query: str) -> tuple[bool, str]:
    """
    Main interface for guardrails.
    Returns:
        (triggered: bool, response_content: str)
    """
    # 1. Check for greetings first
    greeting_response = check_greetings(query)
    if greeting_response:
        return True, greeting_response
        
    # 2. Check for sensitive topics
    sensitive_response = check_sensitive_topics(query)
    if sensitive_response:
        return True, sensitive_response
        
    return False, ""
