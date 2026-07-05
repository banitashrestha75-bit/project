import re
import math
import string
import database

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


class BM25:
    """Pure-Python implementation of BM25 (Best Matching 25) ranking algorithm."""
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents  # List of dicts, must contain 'content'
        self.corpus_size = len(documents)
        self.avg_doc_len = 0
        self.doc_lengths = []
        self.doc_term_freqs = []    # List of term-frequency dicts for each doc
        self.df = {}                # Document frequencies for words
        self.idf = {}               # Inverse document frequencies for words
        
        if self.corpus_size == 0:
            return
            
        total_length = 0
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len
            
            # Count term frequencies
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            
            # Update document frequency
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avg_doc_len = total_length / self.corpus_size
        self._calc_idf()
        
    def _tokenize(self, text: str) -> list[str]:
        """Convert to lowercase, replace punctuation with spaces, and split."""
        text = text.lower()
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        return text.split()
        
    def _calc_idf(self):
        """Calculate IDF for each unique term."""
        for term, freq in self.df.items():
            # BM25 standard IDF formula
            numerator = self.corpus_size - freq + 0.5
            denominator = freq + 0.5
            # Use max(0.0001) to prevent log of negative values
            self.idf[term] = math.log(max(0.0001, numerator / denominator) + 1.0)
            
    def get_scores(self, query: str) -> list[float]:
        """Calculates BM25 relevance score for each document against the query."""
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
                
                # Formula adjustment for document length
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf_val * (numerator / denominator)
                
        return scores


def retrieve_relevant_chunks(query: str, user_id: int, top_k: int = 4, threshold: float = 0.1) -> list[dict]:
    """
    Retrieves the most relevant document chunks for a specific user query.
    Returns chunks sorted by descending relevance.
    """
    # Get all saved chunks for the user
    chunks = database.get_user_chunks(user_id)
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
            scored_chunks.append(chunk_copy)
            
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]
