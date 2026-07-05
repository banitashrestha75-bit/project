# Streamlit RAG Chatbot with Authentication & Guardrails

This is a comprehensive, production-ready Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Groq, LlamaParse, Tavily, and SQLite.

## Features
- **User Authentication**: Login, Sign up, and Session management with SQLite database.
- **Multi-Document Parsing**: Supports PDF, TXT, MD, and JSON document uploads. Uses **LlamaParse** for complex layout PDF parsing, with a fallback to `pypdf`.
- **Saved Chunks Viewer**: View parsed document chunks in an interactive table/list, with a button to save/index them.
- **Pure-Python BM25 Retrieval**: Fast and lightweight search across indexed chunks without heavy local embedding models or database dependencies.
- **Tavily Web Search Fallback**: Automatically searches the web via Tavily if query is not answered in uploaded documents.
- **Guardrails**: Intercepts sensitive topics (self-harm, violence, private PII) and handles normal greetings gracefully.
- **Premium Aesthetics**: Dark theme, modern layouts, micro-animations, and glassmorphism styling.
- **Streamlit Cloud Ready**: Easy deployment with standard environment configurations.

## Setup Instructions

1. **Clone or Copy Files** to your local system.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**:
   Create a `.env` file based on `.env.example` and fill in your keys:
   ```env
   GROQ_API_KEY=your_groq_key
   LLAMA_CLOUD_API_KEY=your_llamaparse_key
   TAVILY_API_KEY=your_tavily_key
   ```
5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment
To deploy on Streamlit Cloud:
1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and select the repo.
3. In Advanced Settings, add the keys under Secrets:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   LLAMA_CLOUD_API_KEY = "your_llamaparse_key"
   TAVILY_API_KEY = "your_tavily_key"
   ```
