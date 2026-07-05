import re
import math
import string
import database
from logger_setup import logger

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """
    Splits text recursively. It tries to split on paragraphs first, 
    then sentences, then words, respecting chunk_size and chunk_overlap.
    """
    if not text or not text.strip():
        return []
        
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Check if adding this paragraph exceeds chunk size
        if current_length + len(paragraph) + (2 if current_chunk else 0) <= chunk_size:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + (2 if len(current_chunk) > 1 else 0)
        else:
            # Save the current chunk if it has content
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Retain some sentences from the end for overlap
                overlap_text = ""
                last_paragraph = current_chunk[-1]
                sentences = re.split(r'(?<=[.!?]) +', last_paragraph)
                overlap_sentences = []
                overlap_len = 0
                for sent in reversed(sentences):
                    if overlap_len + len(sent) + (1 if overlap_sentences else 0) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_len += len(sent) + (1 if len(overlap_sentences) > 1 else 0)
                    else:
                        break
                current_chunk = overlap_sentences if overlap_sentences else []
                current_length = overlap_len
            
            # Handle paragraphs larger than the maximum chunk size
            if len(paragraph) > chunk_size:
                sentences = re.split(r'(?<=[.!?]) +', paragraph)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_length + len(sent) + (1 if current_chunk else 0) <= chunk_size:
                        current_chunk.append(sent)
                        current_length += len(sent) + (1 if len(current_chunk) > 1 else 0)
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        # Start new chunk with overlap
                        overlap_sentences = []
                        overlap_len = 0
                        if current_chunk:
                            # overlap from the end of the previous sentence chunk
                            last_sent = current_chunk[-1]
                            if len(last_sent) <= chunk_overlap:
                                overlap_sentences = [last_sent]
                                overlap_len = len(last_sent)
                        current_chunk = overlap_sentences + [sent]
                        current_length = overlap_len + len(sent) + (1 if overlap_sentences else 0)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + (2 if len(current_chunk) > 1 else 0)
                
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

# --- Vector Embedding Generation via Groq ---
def generate_embeddings(texts: list[str], client) -> list[list[float]]:
    """Generates embedding vectors for a list of texts using Groq embeddings API."""
    if not texts:
        return []
        
    embeddings = []
    # Groq embeddings limit: let's batch in groups of 16 to avoid payload size issues
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model="nomic-embed-text-v1.5"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings via Groq: {e}")
            # Fallback to zero vectors if API fails so the execution doesn't crash completely
            # nomic-embed-text-v1.5 dimension is 768
            zero_vec = [0.0] * 768
            embeddings.extend([zero_vec] * len(batch))
            
    return embeddings

def generate_single_embedding(text: str, client) -> list[float]:
    """Generates an embedding vector for a single text string."""
    res = generate_embeddings([text], client)
    return res[0] if res else [0.0] * 768

# --- Cosine Similarity Helpers ---
def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(a * a for a in v))

def cosine_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

# --- Vector Search ---
def retrieve_vector_chunks(query: str, user_id: int, client, top_k: int = 4, threshold: float = 0.1) -> list[dict]:
    """Retrieves document chunks using semantic vector cosine similarity."""
    chunks = database.get_user_vector_chunks(user_id)
    if not chunks:
        logger.info(f"No vector chunks found in DB for user ID {user_id}")
        return []
        
    # Generate embedding for user query
    query_emb = generate_single_embedding(query, client)
    
    scored_chunks = []
    for chunk in chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        if sim >= threshold:
            chunk_copy = dict(chunk)
            chunk_copy["score"] = sim
            # Clean embedding from output to save context tokens and memory
            if "embedding" in chunk_copy:
                del chunk_copy["embedding"]
            scored_chunks.append(chunk_copy)
            
    # Sort by similarity score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"Vector search retrieved {len(scored_chunks[:top_k])} chunks for query: '{query}'")
    return scored_chunks[:top_k]

# --- BM25 Keyword Search Fallback/Alternative ---
class BM25:
    """Pure-Python implementation of BM25 (Best Matching 25) ranking algorithm."""
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus_size = len(documents)
        self.avg_doc_len = 0
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.df = {}
        self.idf = {}
        
        if self.corpus_size == 0:
            return
            
        total_length = 0
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len
            
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avg_doc_len = total_length / self.corpus_size
        self._calc_idf()
        
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        return text.split()
        
    def _calc_idf(self):
        for term, freq in self.df.items():
            numerator = self.corpus_size - freq + 0.5
            denominator = freq + 0.5
            self.idf[term] = math.log(max(0.0001, numerator / denominator) + 1.0)
            
    def get_scores(self, query: str) -> list[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        if self.corpus_size == 0:
            return scores
            
        for q_term in query_tokens:
            if q_term not in self.idf:
                continue
            idf_val = self.idf[q_term]
            
            for idx in range(self.corpus_size):
                tf = self.doc_term_freqs[idx].get(q_term, 0)
                doc_len = self.doc_lengths[idx]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf_val * (numerator / denominator)
                
        return scores

def retrieve_bm25_chunks(query: str, user_id: int, top_k: int = 4, threshold: float = 0.1) -> list[dict]:
    """Retrieves document chunks using keyword-based BM25 matching."""
    chunks = database.get_user_vector_chunks(user_id)
    if not chunks:
        return []
        
    bm25 = BM25(chunks)
    scores = bm25.get_scores(query)
    
    scored_chunks = []
    for idx, chunk in enumerate(chunks):
        score = scores[idx]
        if score >= threshold:
            chunk_copy = dict(chunk)
            chunk_copy["score"] = score
            if "embedding" in chunk_copy:
                del chunk_copy["embedding"]
            scored_chunks.append(chunk_copy)
            
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]
