import os
import json
import urllib.request
import streamlit as st

# Try importing Tavily Client
try:
    from tavily import TavilyClient
    HAS_TAVILY_CLIENT = True
except ImportError:
    HAS_TAVILY_CLIENT = False

def get_tavily_key() -> str:
    """Helper to fetch Tavily API key from environment variables or Streamlit secrets."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        try:
            if "TAVILY_API_KEY" in st.secrets:
                tavily_key = st.secrets["TAVILY_API_KEY"]
        except Exception:
            pass
    return tavily_key

def search_tavily_api(query: str, api_key: str) -> dict:
    """Performs search using raw HTTP request to Tavily endpoint (no external SDK required)."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 4
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = response.read().decode("utf-8")
            return json.loads(res_data)
    except Exception as e:
        raise RuntimeError(f"Tavily API raw HTTP request failed: {e}")

def search_tavily_sdk(query: str, api_key: str) -> dict:
    """Performs search using Tavily Python SDK."""
    if not HAS_TAVILY_CLIENT:
        raise ImportError("tavily-python library is not installed.")
    
    client = TavilyClient(api_key=api_key)
    # The client.search returns a dictionary containing results
    return client.search(query=query, search_depth="basic", max_results=4, include_answer=True)

def web_search(query: str) -> tuple[list[dict], str]:
    """
    Performs web search using Tavily.
    Returns:
        (results: list[dict], direct_answer: str)
    """
    api_key = get_tavily_key()
    if not api_key:
        return [], "Tavily API key is not configured. Web search is unavailable."
        
    try:
        if HAS_TAVILY_CLIENT:
            response = search_tavily_sdk(query, api_key)
        else:
            response = search_tavily_api(query, api_key)
            
        results = response.get("results", [])
        direct_answer = response.get("answer", "")
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title", "No Title"),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            })
            
        return formatted_results, direct_answer
        
    except Exception as e:
        # Fallback to direct HTTP request if SDK fails
        try:
            response = search_tavily_api(query, api_key)
            results = response.get("results", [])
            direct_answer = response.get("answer", "")
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", "No Title"),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
            return formatted_results, direct_answer
        except Exception as raw_e:
            return [], f"Web search failed. Errors: [SDK: {e}], [HTTP: {raw_e}]"
